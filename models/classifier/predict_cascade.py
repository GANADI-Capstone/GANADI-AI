"""
강아지 캐스케이드 추론 (Binary → Group → Subgroup) + TTA.

각막: 세부 모델 없음 → '각막계 질환 의심' + 수의사 상담 권장.
안검·기타: subgroup 신뢰도 < 0.8 → '{부위}계 질환 의심'만 출력.

실행:
  python models/classifier/predict_cascade.py \\
    --image path/to/image.jpg \\
    --checkpoint-dir models/classifier/checkpoints \\
    --use-tta
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.classifier.dataset import get_transforms
from models.classifier.dataset_group import get_group_map
from models.classifier.model_binary import create_binary_model
from models.classifier.model_group import create_group_model
from models.classifier.model_subgroup import create_subgroup_model
from models.classifier.model_subgroup_cbam import create_subgroup_cbam_model
from models.classifier.predict_tta import tta_logits

# 세부 subgroup 모델을 쓰는 부위 (각막 제외)
SUBGROUP_INFERENCE_GROUPS = frozenset({"안검", "기타"})

SUBGROUP_CKPT_SLUG = {
    "안검": "eyelid",
    "기타": "etc",
}

CORNEA_GROUP = "각막"
CORNEA_MESSAGE = "각막계 질환이 의심됩니다. 수의사 상담 권장."
DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("CASCADE_CONF_THRESHOLD", "0.8"))


def group_suspicion_label(group_name: str) -> str:
    return f"{group_name}계 질환 의심"


class DogCascadePredictor:
    """강아지 3단계 캐스케이드 + TTA."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        use_tta: bool = True,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.use_tta = use_tta
        self.confidence_threshold = confidence_threshold
        self.device = device or self._default_device()
        self.group_map = get_group_map("dog")

        self.binary_model = self._load_binary()
        self.group_model = self._load_group()
        self.subgroup_models: Dict[str, torch.nn.Module] = {}
        self.subgroup_meta: Dict[str, dict] = {}

        for group_name, slug in SUBGROUP_CKPT_SLUG.items():
            model, meta = self._load_subgroup(group_name, slug)
            if model is not None:
                self.subgroup_models[group_name] = model
                self.subgroup_meta[group_name] = meta

        print(
            f"✓ Cascade Predictor (device={self.device}, TTA={'ON' if use_tta else 'OFF'}, "
            f"conf≥{self.confidence_threshold:.2f} for detail)"
        )

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_state(self, path: str, model: torch.nn.Module) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return ckpt

    def _load_binary(self):
        path = os.path.join(self.checkpoint_dir, "dog_binary_best.pth")
        model = create_binary_model("dog", pretrained=False).to(self.device)
        self._load_state(path, model)
        return model

    def _load_group(self):
        path = os.path.join(self.checkpoint_dir, "dog_group_best.pth")
        model = create_group_model("dog", pretrained=False).to(self.device)
        self._load_state(path, model)
        return model

    def _load_subgroup(self, group_name: str, slug: str):
        path = os.path.join(self.checkpoint_dir, f"dog_{slug}_best.pth")
        if not os.path.isfile(path):
            print(f"⚠ Subgroup 체크포인트 없음: {path}")
            return None, {}
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        class_names = ckpt.get("class_names") or self.group_map[group_name]
        num_classes = len(class_names)
        img_size = ckpt.get("img_size", 300)

        use_cbam = group_name == "안검"
        if use_cbam:
            model = create_subgroup_cbam_model(
                num_classes=num_classes,
                class_names=class_names,
                group_name=group_name,
                pretrained=False,
                use_projection=False,
            )
        else:
            model = create_subgroup_model(
                num_classes=num_classes,
                class_names=class_names,
                group_name=group_name,
                pretrained=False,
            )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(self.device)
        return model, {"class_names": class_names, "img_size": img_size}

    def _preprocess(self, image_path: str, img_size: int) -> torch.Tensor:
        image = np.array(Image.open(image_path).convert("RGB"))
        transform = get_transforms(img_size, is_training=False)
        tensor = transform(image=image)["image"].unsqueeze(0)
        return tensor.to(self.device)

    def _infer_logits(self, model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        if self.use_tta:
            return tta_logits(model, tensor)
        with torch.no_grad():
            return model(tensor)

    def _apply_group_suspicion(
        self,
        result: Dict,
        group_name: str,
        confidence: float,
        *,
        message: Optional[str] = None,
    ) -> Dict:
        result["disease"] = group_suspicion_label(group_name)
        result["disease_confidence"] = confidence
        result["detail_level"] = "group_suspicion"
        result["message"] = message or group_suspicion_label(group_name)
        result["specific_disease"] = None
        return result

    def _apply_specific_disease(
        self,
        result: Dict,
        disease: str,
        confidence: float,
    ) -> Dict:
        result["disease"] = disease
        result["disease_confidence"] = confidence
        result["detail_level"] = "specific"
        result["message"] = None
        result["specific_disease"] = disease
        return result

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        tensor300 = self._preprocess(image_path, 300)

        bin_logits = self._infer_logits(self.binary_model, tensor300)
        bin_probs = F.softmax(bin_logits, dim=1)[0]
        is_abnormal = int(bin_probs.argmax().item()) == 1

        result: Dict = {
            "is_normal": not is_abnormal,
            "binary_probs": bin_probs.cpu().tolist(),
            "group": None,
            "group_confidence": 0.0,
            "disease": None,
            "disease_confidence": 0.0,
            "detail_level": None,
            "message": None,
            "specific_disease": None,
            "pipeline": "dog_cascade",
            "tta": self.use_tta,
            "confidence_threshold": self.confidence_threshold,
        }
        if not is_abnormal:
            return result

        grp_logits = self._infer_logits(self.group_model, tensor300)
        grp_probs = F.softmax(grp_logits, dim=1)[0]
        grp_idx = int(grp_probs.argmax().item())
        group_name = self.group_model.class_names[grp_idx]
        group_conf = float(grp_probs[grp_idx].item())
        result["group"] = group_name
        result["group_confidence"] = group_conf

        # 각막: 세부 모델 없음 — 항상 부위 수준 의심만
        if group_name == CORNEA_GROUP:
            return self._apply_group_suspicion(
                result, group_name, group_conf, message=CORNEA_MESSAGE,
            )

        diseases = self.group_map[group_name]

        # 결막·수정체 등 질환 1개 부위 → 그룹 결과로 질환 확정
        if len(diseases) == 1:
            return self._apply_specific_disease(result, diseases[0], group_conf)

        # 안검·기타: subgroup (신뢰도 미달 시 부위 수준만)
        if group_name not in SUBGROUP_INFERENCE_GROUPS:
            return self._apply_specific_disease(result, diseases[0], group_conf)

        sub_model = self.subgroup_models.get(group_name)
        if sub_model is None:
            return self._apply_group_suspicion(result, group_name, group_conf)

        meta = self.subgroup_meta[group_name]
        img_size = meta.get("img_size", 300)
        tensor_sub = self._preprocess(image_path, img_size) if img_size != 300 else tensor300

        sub_logits = self._infer_logits(sub_model, tensor_sub)
        sub_probs = F.softmax(sub_logits, dim=1)[0]
        sub_idx = int(sub_probs.argmax().item())
        sub_conf = float(sub_probs[sub_idx].item())
        specific = meta["class_names"][sub_idx]

        if sub_conf < self.confidence_threshold:
            return self._apply_group_suspicion(result, group_name, sub_conf)

        return self._apply_specific_disease(result, specific, sub_conf)


def main():
    parser = argparse.ArgumentParser(description="강아지 캐스케이드 추론 + TTA")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint-dir", default="models/classifier/checkpoints")
    parser.add_argument("--use-tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="안검·기타 세부 질환 출력 최소 신뢰도 (기본 0.8)",
    )
    args = parser.parse_args()

    use_tta = args.use_tta and not args.no_tta
    predictor = DogCascadePredictor(
        args.checkpoint_dir,
        use_tta=use_tta,
        confidence_threshold=args.conf_threshold,
    )
    result = predictor.predict(args.image)
    if result.get("message"):
        print(result["message"])
    print(result)


if __name__ == "__main__":
    main()

"""
멀티태스크 모델 GradCAM 감사 — 특정 헤드 P(비정상) 주목 영역 시각화.

백내장 헤드 과예측(아티팩트 vs 실제 병변) 가설 검증용.
TL random split Val (SPLIT_SEED=42, VAL_RATIO=0.2) 샘플을 필터링해 heatmap 저장.

실행 (프로젝트 루트, Colab A100):
  python models/classifier/gradcam_audit.py \\
    --animal dog --head 백내장 --mode misclassified \\
    --true-disease 핵경화 --n 20 --out gradcam_audit_out/

  python models/classifier/gradcam_audit.py \\
    --animal dog --head 백내장 --mode correct --n 20

환경변수:
  SPLIT_SEED=42  VAL_RATIO=0.2  CHECKPOINT=  BATCH_SIZE=32  IMG_SIZE=300
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.dataset_random_split import (
    RandomSplitEyeDataset,
    create_random_split_dataloaders,
)
from models.classifier.eval_multitask_topk import (
    _active_disease_and_label,
    _rank_diseases_by_abnormal_prob,
    head_abnormal_probability,
    resolve_checkpoint,
)
from models.classifier.inference_multitask import extract_state_dict
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers

IMG_SIZE_DEFAULT = 300

HEAD_ALIASES: Dict[str, str] = {
    "cataract": "백내장",
    "백내장": "백내장",
    "nuclear_sclerosis": "핵경화",
    "핵경화": "핵경화",
    "conjunctivitis": "결막염",
    "결막염": "결막염",
    "kcs": "유루증",
    "유루증": "유루증",
    "entropion": "안검내반증",
    "안검내반증": "안검내반증",
}


@dataclass
class AuditSample:
    local_idx: int
    img_path: str
    true_disease: str
    true_label: int
    pred_disease: str
    pred_conf: float
    head_abnormal: float


class MultiTaskHeadWrapper(nn.Module):
    """단일 질환 헤드 로짓만 반환 (GradCAM용)."""

    def __init__(self, model: nn.Module, disease_name: str):
        super().__init__()
        self.model = model
        self.disease_name = disease_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[self.disease_name]


class HeadAbnormalProbTarget:
    """GradCAM target: P(비정상) = 1 - softmax(logits)[0]."""

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        probs = F.softmax(model_output, dim=-1)
        return 1.0 - probs[:, 0]


def resolve_head(name: str, diseases: Sequence[str]) -> str:
    key = name.strip()
    resolved = HEAD_ALIASES.get(key, key)
    if resolved not in diseases:
        raise ValueError(
            f"알 수 없는 --head '{name}' (resolved='{resolved}'). "
            f"가능: {list(diseases)}"
        )
    return resolved


def resolve_gradcam_target_layers(model: nn.Module) -> List[nn.Module]:
    """EfficientNet-B3 백본 마지막 conv block 자동 탐색."""
    backbone = model.backbone

    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        last_block = backbone.blocks[-1]
        inner = last_block[-1] if hasattr(last_block, "__getitem__") else last_block
        convs = [m for m in inner.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            layer = convs[-1]
            print(
                f"  GradCAM target: backbone.blocks[-1][-1] Conv2d "
                f"{tuple(layer.weight.shape)}"
            )
            return [layer]
        print("  GradCAM target: backbone.blocks[-1][-1] (fallback module)")
        return [inner]

    if hasattr(backbone, "conv_head"):
        print("  GradCAM target: backbone.conv_head (fallback)")
        return [backbone.conv_head]

    raise RuntimeError("EfficientNet-B3 target layer를 찾지 못했습니다.")


def load_multitask_checkpoint(
    animal_type: str,
    checkpoint: Path,
    device: torch.device,
) -> nn.Module:
    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  checkpoint: {checkpoint}")
    print(
        f"  epoch={ckpt.get('epoch')}  task={ckpt.get('task')}  "
        f"val_acc_mean={ckpt.get('val_acc_mean')}"
    )
    return model


def get_sample_meta(
    val_ds: RandomSplitEyeDataset,
    local_idx: int,
) -> Tuple[str, Dict[str, int]]:
    base_idx = val_ds.indices[local_idx]
    return val_ds.base.samples[base_idx]


def label_to_name(label_map: Dict[str, Dict[str, int]], disease: str, idx: int) -> str:
    reverse = {v: k for k, v in label_map[disease].items()}
    return reverse.get(idx, str(idx))


def matches_mode(
    *,
    mode: str,
    head: str,
    true_disease: Optional[str],
    gt_disease: Optional[str],
    gt_label: int,
    pred_top1: str,
) -> bool:
    if gt_disease is None or gt_label <= 0:
        return False

    if mode == "correct":
        return gt_disease == head and pred_top1 == head

    if mode == "misclassified":
        if not true_disease:
            raise ValueError("misclassified 모드는 --true-disease 가 필요합니다.")
        return gt_disease == true_disease and pred_top1 == head and gt_disease != head

    raise ValueError(f"mode는 correct | misclassified: {mode}")


@torch.no_grad()
def collect_candidates(
    model: nn.Module,
    val_ds: RandomSplitEyeDataset,
    val_loader: DataLoader,
    diseases: List[str],
    head: str,
    mode: str,
    true_disease: Optional[str],
    max_n: int,
    device: torch.device,
) -> List[AuditSample]:
    """Val 전체 batch 추론 -> 조건에 맞는 샘플 인덱스 수집."""
    candidates: List[AuditSample] = []
    local_idx = 0

    for images, labels in tqdm(val_loader, desc=f"Scan val ({mode})"):
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(val_ds):
                break

            gt_disease, gt_label = _active_disease_and_label(labels, i, diseases)
            ranked = _rank_diseases_by_abnormal_prob(outputs, i, diseases)
            pred_top1, pred_conf = ranked[0]

            if matches_mode(
                mode=mode,
                head=head,
                true_disease=true_disease,
                gt_disease=gt_disease,
                gt_label=gt_label,
                pred_top1=pred_top1,
            ):
                img_path, _ = get_sample_meta(val_ds, local_idx)
                head_abn = head_abnormal_probability(outputs[head][i])
                candidates.append(
                    AuditSample(
                        local_idx=local_idx,
                        img_path=img_path,
                        true_disease=gt_disease,
                        true_label=gt_label,
                        pred_disease=pred_top1,
                        pred_conf=pred_conf,
                        head_abnormal=head_abn,
                    )
                )
                if len(candidates) >= max_n:
                    return candidates

            local_idx += 1

    return candidates


def load_rgb_and_tensor(
    img_path: str,
    img_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    """시각화용 RGB [H,W,3] 0~1 + 모델 입력 tensor [1,3,H,W]."""
    pil = Image.open(img_path).convert("RGB")
    rgb = np.array(pil.resize((img_size, img_size)), dtype=np.float32) / 255.0

    image_np = np.array(pil)
    from models.classifier.dataset import get_transforms

    transformed = get_transforms(img_size=img_size, is_training=False)(image=image_np)
    tensor = transformed["image"].unsqueeze(0).to(device)
    return rgb, tensor


def save_side_by_side(
    rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """원본 | GradCAM 나란히 저장."""
    h, w = rgb.shape[:2]
    left = (rgb * 255).astype(np.uint8)
    right = overlay_rgb
    if right.shape[:2] != (h, w):
        right = cv2.resize(right, (w, h))

    gap = 4
    canvas = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 255
    canvas[:, :w] = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    canvas[:, w + gap :] = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

    if title:
        cv2.putText(
            canvas,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_gradcam_audit(
    *,
    animal: str,
    head: str,
    mode: str,
    true_disease: Optional[str],
    n: int,
    out_dir: Path,
    checkpoint: Optional[Path],
    img_size: int,
    batch_size: int,
    device: str,
) -> None:
    dev = torch.device(device)
    checkpoint = checkpoint or resolve_checkpoint(animal)

    print("=" * 64)
    print(f"GradCAM Audit — {animal.upper()}  head={head}  mode={mode}")
    print("=" * 64)

    model = load_multitask_checkpoint(animal, checkpoint, dev)
    diseases = model.get_disease_names()
    label_map = model.get_label_map()
    head = resolve_head(head, diseases)
    if true_disease:
        true_disease = resolve_head(true_disease, diseases)

    target_layers = resolve_gradcam_target_layers(model)
    wrapped = MultiTaskHeadWrapper(model, head)
    grad_cam = GradCAM(model=wrapped, target_layers=target_layers)
    cam_target = HeadAbnormalProbTarget()

    _, val_loader, _, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal,
        batch_size=resolve_batch_size(batch_size),
        img_size=img_size,
        num_workers=resolve_num_workers(4),
        use_sampler=False,
        mode="multitask",
    )

    print(f"  Val size: {len(val_ds):,}  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    if true_disease:
        print(f"  true-disease filter: {true_disease}")

    candidates = collect_candidates(
        model,
        val_ds,
        val_loader,
        diseases,
        head,
        mode,
        true_disease,
        n,
        dev,
    )

    if not candidates:
        print(f"\n⚠ 조건에 맞는 샘플 0건 (mode={mode}, head={head})")
        return

    print(f"\n✓ {len(candidates)}건 GradCAM 생성 (--head={head}, P(비정상) target)")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: List[str] = []

    for k, sample in enumerate(tqdm(candidates, desc="GradCAM")):
        rgb, tensor = load_rgb_and_tensor(sample.img_path, img_size, dev)

        grayscale_cam = grad_cam(input_tensor=tensor, targets=[cam_target])
        cam = grayscale_cam[0]
        overlay = show_cam_on_image(rgb, cam, use_rgb=True)

        true_name = label_to_name(label_map, sample.true_disease, sample.true_label)
        conf_str = f"{sample.pred_conf:.3f}".replace(".", "p")
        fname = (
            f"{k:03d}_{sample.true_disease}_gt{true_name}_"
            f"pred{sample.pred_disease}_conf{conf_str}_"
            f"headAbn{sample.head_abnormal:.3f}.png"
        )
        out_path = out_dir / fname

        title = (
            f"GT:{sample.true_disease}({true_name}) "
            f"Top1:{sample.pred_disease}({sample.pred_conf:.2f}) "
            f"{head} P(abn)={sample.head_abnormal:.2f}"
        )
        save_side_by_side(rgb, overlay, out_path, title=title)

        summary_lines.append(f"{fname}\n  path={sample.img_path}\n  {title}\n")

    summary_path = out_dir / "audit_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"animal={animal}",
                f"head={head}",
                f"mode={mode}",
                f"true_disease={true_disease}",
                f"n={len(candidates)}",
                f"checkpoint={checkpoint}",
                f"split_meta={split_meta}",
                "",
                *summary_lines,
            ]
        ),
        encoding="utf-8",
    )

    print(f"\n✓ 저장: {out_dir}/ ({len(candidates)} images)")
    print(f"✓ 요약: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="멀티태스크 GradCAM 감사 (헤드별 P(비정상) 주목 영역)",
    )
    parser.add_argument(
        "--animal",
        default=os.environ.get("ANIMAL_TYPE", "dog"),
        choices=["dog", "cat"],
    )
    parser.add_argument(
        "--head",
        required=True,
        help="GradCAM 대상 헤드 (예: 백내장, cataract)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["correct", "misclassified"],
        help="correct=해당 head 정답, misclassified=다른 질환->head 오분류",
    )
    parser.add_argument(
        "--true-disease",
        default=None,
        help="misclassified 모드: 실제 GT 질환 (예: 핵경화)",
    )
    parser.add_argument("--n", type=int, default=20, help="저장 샘플 수")
    parser.add_argument("--out", type=str, default="gradcam_audit_out")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--img-size",
        type=int,
        default=int(os.environ.get("IMG_SIZE", str(IMG_SIZE_DEFAULT))),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "32")),
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "") or get_device(),
        help="cuda | mps | cpu (기본: auto)",
    )

    args = parser.parse_args()

    if args.mode == "misclassified" and not args.true_disease:
        parser.error("misclassified 모드는 --true-disease 가 필요합니다.")

    ckpt = Path(args.checkpoint) if args.checkpoint else None

    run_gradcam_audit(
        animal=args.animal.lower(),
        head=args.head,
        mode=args.mode,
        true_disease=args.true_disease,
        n=args.n,
        out_dir=Path(args.out),
        checkpoint=ckpt,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

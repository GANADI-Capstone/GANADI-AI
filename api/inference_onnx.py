"""
ONNX Runtime 멀티태스크 추론 (PyTorch inference_multitask 와 동일 인터페이스).

INFERENCE_BACKEND=onnx | onnx_int8
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.classifier.inference_multitask import (
    resolve_checkpoint_dir,
    resolve_model_version,
    run_multitask_inference,
)
from models.classifier.model import DOG_DISEASE_HEADS, CAT_DISEASE_HEADS, DOG_LABEL_MAP, CAT_LABEL_MAP

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "pytorch"


def resolve_inference_backend() -> str:
    return os.environ.get("INFERENCE_BACKEND", DEFAULT_BACKEND).strip().lower() or DEFAULT_BACKEND


def uses_onnx_backend(backend: Optional[str] = None) -> bool:
    b = (backend or resolve_inference_backend()).lower()
    return b.startswith("onnx")


def onnx_path_for(animal_type: str, *, int8: bool = False) -> Path:
    animal_type = animal_type.lower()
    base = resolve_checkpoint_dir()
    version = resolve_model_version()
    if version == "random_split":
        suffix = "_int8" if int8 else ""
        return base / f"{animal_type}_best_random_split{suffix}.onnx"
    suffix = "_int8" if int8 else ""
    return base / f"{animal_type}_best_{version}{suffix}.onnx"


def get_ort_providers() -> List[str]:
    """CUDA → CoreML → CPU 우선순위 (환경변수 ORT_PROVIDERS 로 override 가능)."""
    import onnxruntime as ort

    override = os.environ.get("ORT_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]

    available = set(ort.get_available_providers())
    preferred: List[str] = []
    for p in (
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ):
        if p in available:
            preferred.append(p)
    if not preferred:
        preferred = ["CPUExecutionProvider"]
    return preferred


def create_ort_session(path: Path) -> Any:
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = get_ort_providers()
    try:
        return ort.InferenceSession(
            str(path), sess_options=sess_options, providers=providers,
        )
    except Exception as exc:
        if providers != ["CPUExecutionProvider"]:
            logger.warning("ORT providers %s 실패 (%s) → CPU fallback", providers, exc)
            return ort.InferenceSession(
                str(path), sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        raise


class OnnxMultiTaskModel:
    """run_multitask_inference 와 호환되는 ONNX 래퍼."""

    def __init__(
        self,
        animal_type: str,
        session: Any,
        disease_names: List[str],
        label_map: Dict,
        model_path: Path,
    ):
        self.animal_type = animal_type.lower()
        self.session = session
        self._disease_names = disease_names
        self._label_map = label_map
        self.model_path = model_path
        self._input_name = session.get_inputs()[0].name
        self._output_names = [f"head_{d}" for d in disease_names]

    def get_disease_names(self) -> List[str]:
        return self._disease_names

    def get_label_map(self) -> Dict:
        return self._label_map

    def get_num_classes(self, disease: str) -> int:
        heads = DOG_DISEASE_HEADS if self.animal_type == "dog" else CAT_DISEASE_HEADS
        return heads.get(disease, 0)

    def eval(self) -> "OnnxMultiTaskModel":
        return self

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        np_in = x.detach().cpu().numpy().astype(np.float32)
        ort_outs = self.session.run(self._output_names, {self._input_name: np_in})
        return {
            d: torch.from_numpy(o)
            for d, o in zip(self._disease_names, ort_outs)
        }


def load_onnx_model(
    animal_type: str,
    *,
    int8: bool = False,
    checkpoint_override: Optional[str] = None,
) -> Tuple[OnnxMultiTaskModel, Path]:
    animal_type = animal_type.lower()
    path = Path(checkpoint_override) if checkpoint_override else onnx_path_for(animal_type, int8=int8)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX 모델 없음: {path}")

    import onnxruntime as ort

    session = create_ort_session(path)

    label_map = DOG_LABEL_MAP if animal_type == "dog" else CAT_LABEL_MAP
    disease_names = list((DOG_DISEASE_HEADS if animal_type == "dog" else CAT_DISEASE_HEADS).keys())

    model = OnnxMultiTaskModel(animal_type, session, disease_names, label_map, path)
    logger.info(
        "ONNX 로드: %s providers=%s int8=%s",
        path.name,
        session.get_providers(),
        int8,
    )
    return model, path


def run_onnx_inference(
    model: OnnxMultiTaskModel,
    input_tensor: torch.Tensor,
    *,
    top_k: int = 3,
    abnormal_threshold: float = 0.5,
    clear_threshold: float = 0.7,
    device_meta: Optional[str] = None,
) -> Dict[str, Any]:
    """ONNX 추론 — post-processing은 inference_multitask 공유."""
    return run_multitask_inference(
        model,
        input_tensor,
        device="cpu",
        top_k=top_k,
        abnormal_threshold=abnormal_threshold,
        clear_threshold=clear_threshold,
        device_meta=device_meta,
    )

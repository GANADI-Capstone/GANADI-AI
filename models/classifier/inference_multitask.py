"""
멀티태스크 random split 모델 추론 (AI 서버 / 배치 공용).

Top-K 질환 ranking: 각 헤드 P(비정상) = 1 - P(무).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier.model import create_model

MEDICAL_DEVICES = frozenset({"검안경", "일반카메라"})
SIMPLIFY_DISEASES = frozenset({"백내장", "궤양성각막질환", "비궤양성각막질환"})

DEFAULT_MODEL_VERSION = "random_split"
DEFAULT_CHECKPOINT_DIR = "models/classifier/checkpoints"


def resolve_model_version() -> str:
    return os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION).strip() or DEFAULT_MODEL_VERSION


def resolve_checkpoint_dir() -> Path:
    return Path(os.environ.get("MODEL_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))


def checkpoint_path_for(animal_type: str, version: Optional[str] = None) -> Path:
    """체크포인트 경로 결정.

    MODEL_VERSION:
      random_split (기본) → {animal}_best_random_split.pth
      legacy            → {animal}_best.pth
      기타              → {animal}_best_{version}.pth
    """
    animal_type = animal_type.lower()
    version = (version or resolve_model_version()).strip()
    base = resolve_checkpoint_dir()

    if version in ("legacy", "best", "v1"):
        return base / f"{animal_type}_best.pth"
    if version == "random_split":
        return base / f"{animal_type}_best_random_split.pth"
    return base / f"{animal_type}_best_{version}.pth"


def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """EMA / model_state_dict / state_dict 자동 인식."""
    for key in ("model_state_dict", "model_state_ema", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    raise KeyError(
        "체크포인트에 model_state_dict / model_state_ema / state_dict 없음"
    )


def load_multitask_model(
    animal_type: str,
    device: torch.device | str,
    *,
    version: Optional[str] = None,
    checkpoint_override: Optional[str] = None,
) -> Tuple[nn.Module, Path, Dict[str, Any]]:
    path = Path(checkpoint_override) if checkpoint_override else checkpoint_path_for(animal_type, version)
    if not path.is_file():
        raise FileNotFoundError(f"체크포인트 없음: {path}")

    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(ckpt))
    model.to(device)
    model.eval()
    return model, path, ckpt


def head_abnormal_probability(logits: torch.Tensor) -> float:
    """P(비정상) = 1 - P(클래스 0=무). 2/3/4-class 공통."""
    probs = F.softmax(logits, dim=-1)
    return float((1.0 - probs[0]).item())


def _head_display_label(
    disease: str,
    logits: torch.Tensor,
    label_map: Dict[str, Dict[str, int]],
) -> str:
    probs = F.softmax(logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    reverse = {v: k for k, v in label_map[disease].items()}
    label = reverse.get(idx, "무")
    if disease in SIMPLIFY_DISEASES and label != "무":
        return "유"
    return label


@torch.no_grad()
def run_multitask_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    device: torch.device | str,
    top_k: int = 3,
    abnormal_threshold: float = 0.5,
    clear_threshold: float = 0.7,
    device_meta: Optional[str] = None,
) -> Dict[str, Any]:
    """멀티태스크 추론 + Top-K + 레거시 predictions 호환."""
    model.eval()
    input_tensor = input_tensor.to(device)
    outputs = model(input_tensor)

    diseases: List[str] = model.get_disease_names()
    label_map = model.get_label_map()

    abnormal_scores: Dict[str, float] = {}
    legacy_predictions: Dict[str, Dict[str, float | str]] = {}
    any_head_abnormal = False

    for disease in diseases:
        logits = outputs[disease][0]
        abn = head_abnormal_probability(logits)
        abnormal_scores[disease] = abn
        display_label = _head_display_label(disease, logits, label_map)
        if display_label != "무":
            any_head_abnormal = True
        legacy_predictions[disease] = {
            "label": display_label,
            "confidence": round(abn * 100, 1),
        }

    ranked = sorted(abnormal_scores.items(), key=lambda x: (-x[1], x[0]))
    top_k = min(top_k, len(ranked))
    top_diseases = [
        {"disease": name, "confidence": round(score, 4)}
        for name, score in ranked[:top_k]
    ]

    top1_name, top1_score = ranked[0]
    is_normal = not any_head_abnormal and top1_score < abnormal_threshold
    binary_result = "정상" if is_normal else "비정상"

    if is_normal:
        recommendation = (
            "AI 스크리닝상 특이 소견이 두드러지지 않습니다. "
            "이상 증상이 지속되면 수의사 상담을 권장합니다."
        )
    elif top1_score >= clear_threshold:
        recommendation = f"{top1_name} 의심 (AI 스크리닝 소견). 수의사 진료를 권장합니다."
    elif top1_score >= abnormal_threshold:
        recommendation = (
            f"경미한 이상 가능성({top1_name} 등). "
            "정확한 판단은 수의사 상담을 권장합니다."
        )
    else:
        recommendation = (
            "스크리닝 결과가 모호합니다. "
            "정확한 소견은 수의사 상담을 권장합니다."
        )

    device_warning = None
    if device_meta and device_meta.strip() in MEDICAL_DEVICES:
        device_warning = (
            "의료장비(검안경/일반카메라) 촬영으로 보입니다. "
            "본 모델은 스마트폰 촬영 기준으로 학습되어 결과 해석에 주의가 필요합니다."
        )

    main_disease = "" if is_normal else top1_name
    main_confidence = round((top1_score if not is_normal else (1.0 - top1_score)) * 100, 1)

    return {
        # 레거시 (백엔드·리포트·PDF)
        "predictions": legacy_predictions,
        "main_disease": main_disease,
        "main_confidence": main_confidence,
        "is_normal": is_normal,
        # 확장 필드
        "binary_result": binary_result,
        "confidence": round(top1_score if not is_normal else 1.0 - top1_score, 4),
        "top_3_diseases": top_diseases,
        "all_diseases": {k: round(v, 4) for k, v in abnormal_scores.items()},
        "device_warning": device_warning,
        "recommendation": recommendation,
        "model_task": "multitask_random_split",
        "disclaimer": (
            "본 결과는 AI 스크리닝 참고용이며, 정확한 판단은 수의사 진료가 필요합니다."
        ),
    }

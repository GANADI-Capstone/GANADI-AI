"""Test-Time Augmentation (원본 + 좌우반전 평균)."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def tta_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    forward_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    TTA: 원본 + 좌우반전 logits 평균.

    Args:
        model: 분류 모델
        images: [B, C, H, W]
        forward_fn: 기본 model(images). 멀티헤드 등 커스텀 forward 시 지정.
    """
    model.eval()
    fn = forward_fn or model
    logits_orig = fn(images)
    logits_flip = fn(torch.flip(images, dims=[3]))
    if isinstance(logits_orig, dict):
        return {k: (logits_orig[k] + logits_flip[k]) / 2.0 for k in logits_orig}
    return (logits_orig + logits_flip) / 2.0


@torch.no_grad()
def tta_predict_probs(
    model: torch.nn.Module,
    images: torch.Tensor,
    forward_fn: Optional[Callable] = None,
) -> torch.Tensor:
    logits = tta_logits(model, images, forward_fn)
    if isinstance(logits, dict):
        return {k: F.softmax(v, dim=1) for k, v in logits.items()}
    return F.softmax(logits, dim=1)


@torch.no_grad()
def tta_predict_class(
    model: torch.nn.Module,
    images: torch.Tensor,
    forward_fn: Optional[Callable] = None,
) -> torch.Tensor:
    probs = tta_predict_probs(model, images, forward_fn)
    if isinstance(probs, dict):
        return {k: v.argmax(dim=1) for k, v in probs.items()}
    return probs.argmax(dim=1)

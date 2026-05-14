"""멀티태스크 질환 헤드용 손실 함수."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """멀티클래스 Focal Loss.

    FL = alpha_y * (1 - p_t)^gamma * CE(logits, y)
    - alpha: shape [num_classes] (역빈도 등 클래스 가중) 또는 None.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            at = self.alpha[targets]
            focal = at * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_per_disease_losses(
    dataset,
    diseases: list[str],
    loss_type: str,
    *,
    device: str,
    use_class_weights: bool,
    focal_gamma: float,
) -> nn.ModuleDict:
    """질환별 손실 모듈 생성.

    loss_type: ce | weighted_ce | focal
    """
    modules: dict[str, nn.Module] = {}

    for disease in diseases:
        num_classes = len(dataset.label_map[disease])
        class_w: Optional[torch.Tensor] = None
        if use_class_weights:
            cw = dataset.get_class_weights(disease).to(device).float()
            class_w = cw / cw.mean().clamp(min=1e-6)

        if loss_type == "ce":
            modules[disease] = nn.CrossEntropyLoss(weight=class_w)
        elif loss_type == "weighted_ce":
            w_eff = class_w if class_w is not None else torch.ones(
                num_classes, device=device, dtype=torch.float32
            )
            modules[disease] = nn.CrossEntropyLoss(weight=w_eff)
        elif loss_type == "focal":
            modules[disease] = FocalLoss(gamma=focal_gamma, alpha=class_w)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    return nn.ModuleDict(modules)

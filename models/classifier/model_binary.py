"""
EfficientNet-B3 기반 정상 vs 비정상 이진 분류 모델.

질환별 멀티헤드 대신 단일 2-class 헤드 1개.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class BinaryNormalAbnormalModel(nn.Module):
    """EfficientNet-B3 + 단일 이진 분류 헤드 (정상=0, 비정상=1)."""

    NUM_CLASSES = 2

    def __init__(
        self,
        animal_type: str = "cat",
        pretrained: bool = True,
        head_dropout: float = 0.4,
    ):
        super().__init__()
        self.animal_type = animal_type.lower()
        head_dropout = float(max(0.3, min(0.5, head_dropout)))
        head_dropout2 = float(max(0.3, min(0.5, head_dropout + 0.1)))

        if self.animal_type not in ("dog", "cat"):
            raise ValueError(f"animal_type은 'dog' 또는 'cat'이어야 합니다: {animal_type}")

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(head_dropout2),
            nn.Linear(512, self.NUM_CLASSES),
        )

        print(f"✓ {self.animal_type.upper()} 이진 모델 생성 완료")
        print(f"  - 백본: EfficientNet-B3 (feature_dim={self.feature_dim})")
        print(f"  - 헤드: 정상/비정상 2-class, Dropout {head_dropout}/{head_dropout2}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ 백본 freeze (이진 헤드만 학습)")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ 백본 unfreeze (전체 미세조정)")


def create_binary_model(
    animal_type: str = "cat",
    pretrained: bool = True,
    head_dropout: float = 0.4,
) -> BinaryNormalAbnormalModel:
    return BinaryNormalAbnormalModel(
        animal_type=animal_type,
        pretrained=pretrained,
        head_dropout=head_dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

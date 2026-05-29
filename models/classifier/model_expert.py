"""
EfficientNet-B3 기반 질환 분류 전문가(Expert) 모델.

비정상 데이터로 학습한 N-class 단일 헤드 (고양이 5-class).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.classifier.dataset_expert import get_disease_mapping

FEAT_DIM = 512


class ExpertDiseaseModel(nn.Module):
    """EfficientNet-B3 + 질환 단일 분류 헤드."""

    def __init__(
        self,
        animal_type: str = "cat",
        num_classes: int | None = None,
        pretrained: bool = True,
        head_dropout: float = 0.4,
    ):
        super().__init__()
        self.animal_type = animal_type.lower()
        disease_map = get_disease_mapping(self.animal_type)
        self.num_classes = num_classes or len(disease_map)
        self.class_names = [""] * self.num_classes
        for name, idx in disease_map.items():
            if idx < self.num_classes:
                self.class_names[idx] = name

        head_dropout = float(max(0.3, min(0.5, head_dropout)))
        head_dropout2 = float(max(0.3, min(0.5, head_dropout + 0.1)))

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = self.backbone.num_features
        self.feat_dim = FEAT_DIM

        self.head_drop1 = nn.Dropout(head_dropout)
        self.fc1 = nn.Linear(self.feature_dim, FEAT_DIM)
        self.relu = nn.ReLU()
        self.head_drop2 = nn.Dropout(head_dropout2)
        self.fc2 = nn.Linear(FEAT_DIM, self.num_classes)

        print(f"✓ {self.animal_type.upper()} Expert 모델 생성")
        print(f"  - 클래스 수: {self.num_classes}")
        print(f"  - Feature dim: {FEAT_DIM}")
        print(f"  - Dropout: {head_dropout}/{head_dropout2}")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.backbone(x)
        x = self.head_drop1(x)
        features = self.relu(self.fc1(x))
        logits = self.fc2(self.head_drop2(features))
        if return_features:
            # Center Loss 안정화: L2 단위 구면 embedding
            return F.normalize(features, p=2, dim=1), logits
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ 백본 freeze (Expert 헤드만 학습)")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ 백본 unfreeze (전체 미세조정)")


def create_expert_model(
    animal_type: str = "cat",
    pretrained: bool = True,
    head_dropout: float = 0.4,
) -> ExpertDiseaseModel:
    disease_map = get_disease_mapping(animal_type)
    return ExpertDiseaseModel(
        animal_type=animal_type,
        num_classes=len(disease_map),
        pretrained=pretrained,
        head_dropout=head_dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

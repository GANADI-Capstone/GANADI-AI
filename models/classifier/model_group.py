"""EfficientNet-B3 부위 그룹 분류 모델 (강아지 5-class)."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from models.classifier.dataset_group import get_group_label_map


class GroupDiseaseModel(nn.Module):
    """EfficientNet-B3 + 부위 그룹 단일 헤드."""

    HIDDEN_DIM = 512

    def __init__(
        self,
        animal_type: str = "dog",
        num_classes: int | None = None,
        pretrained: bool = True,
        head_dropout: float = 0.4,
    ):
        super().__init__()
        self.animal_type = animal_type.lower()
        label_map = get_group_label_map(self.animal_type)
        self.num_classes = num_classes or len(label_map)
        self.class_names = [""] * self.num_classes
        for name, idx in label_map.items():
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

        self.head_drop1 = nn.Dropout(head_dropout)
        self.fc1 = nn.Linear(self.feature_dim, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.head_drop2 = nn.Dropout(head_dropout2)
        self.fc2 = nn.Linear(self.HIDDEN_DIM, self.num_classes)

        print(f"✓ Group 모델 ({self.animal_type.upper()}, {self.num_classes}-class)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head_drop1(x)
        x = self.relu(self.fc1(x))
        return self.fc2(self.head_drop2(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def create_group_model(
    animal_type: str = "dog",
    pretrained: bool = True,
    head_dropout: float = 0.4,
) -> GroupDiseaseModel:
    return GroupDiseaseModel(
        animal_type=animal_type,
        pretrained=pretrained,
        head_dropout=head_dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

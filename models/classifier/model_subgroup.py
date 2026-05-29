"""EfficientNet-B3 부위별 세부 질환 분류 모델."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class SubgroupDiseaseModel(nn.Module):
    """EfficientNet-B3 + 부위 내 세부 질환 헤드."""

    HIDDEN_DIM = 256

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        group_name: str = "",
        pretrained: bool = True,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.group_name = group_name

        head_dropout = float(max(0.2, min(0.5, head_dropout)))
        head_dropout2 = float(max(0.2, min(0.5, head_dropout + 0.05)))

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
        self.fc2 = nn.Linear(self.HIDDEN_DIM, num_classes)

        print(
            f"✓ Subgroup 모델 [{group_name}] "
            f"{num_classes}-class, hidden={self.HIDDEN_DIM}, dropout={head_dropout}"
        )

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


def create_subgroup_model(
    num_classes: int,
    class_names: list[str],
    group_name: str = "",
    pretrained: bool = True,
    head_dropout: float = 0.3,
) -> SubgroupDiseaseModel:
    return SubgroupDiseaseModel(
        num_classes=num_classes,
        class_names=class_names,
        group_name=group_name,
        pretrained=pretrained,
        head_dropout=head_dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

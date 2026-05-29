"""CBAM + SupCon projection head가 있는 Subgroup 모델."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.classifier.cbam import CBAM


class SubgroupCBAMModel(nn.Module):
    """EfficientNet-B3 (spatial features) + CBAM + 분류 헤드."""

    HIDDEN_DIM = 256
    PROJ_DIM = 128

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        group_name: str = "",
        pretrained: bool = True,
        head_dropout: float = 0.3,
        use_projection: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.group_name = group_name
        self.use_projection = use_projection

        head_dropout = float(max(0.2, min(0.6, head_dropout)))
        head_dropout2 = float(max(0.2, min(0.6, head_dropout + 0.05)))

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.feature_dim = self.backbone.num_features
        self.cbam = CBAM(self.feature_dim)

        self.head_drop1 = nn.Dropout(head_dropout)
        self.fc1 = nn.Linear(self.feature_dim, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.head_drop2 = nn.Dropout(head_dropout2)
        self.fc2 = nn.Linear(self.HIDDEN_DIM, num_classes)

        self.projection = None
        if use_projection:
            self.projection = nn.Linear(self.feature_dim, self.PROJ_DIM)

        extras = []
        if use_projection:
            extras.append(f"SupCon proj {self.feature_dim}→{self.PROJ_DIM}")
        print(
            f"✓ Subgroup CBAM 모델 [{group_name}] "
            f"{num_classes}-class, dropout={head_dropout}"
            + (f" ({', '.join(extras)})" if extras else "")
        )

    def _pool(self, feat_map: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat_map, 1).flatten(1)

    def forward(
        self, x: torch.Tensor, return_projection: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.cbam(self.backbone(x))
        pooled = self._pool(feat_map)

        h = self.head_drop1(pooled)
        h = self.relu(self.fc1(h))
        logits = self.fc2(self.head_drop2(h))

        if return_projection and self.projection is not None:
            proj = F.normalize(self.projection(pooled), dim=1)
            return logits, proj
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def create_subgroup_cbam_model(
    num_classes: int,
    class_names: list[str],
    group_name: str = "",
    pretrained: bool = True,
    head_dropout: float = 0.3,
    use_projection: bool = False,
) -> SubgroupCBAMModel:
    return SubgroupCBAMModel(
        num_classes=num_classes,
        class_names=class_names,
        group_name=group_name,
        pretrained=pretrained,
        head_dropout=head_dropout,
        use_projection=use_projection,
    )

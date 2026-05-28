"""
정상 vs 비정상(통합 이진) 분류용 데이터셋.

기존 EyeDiseaseDataset을 재활용하고, 질환별 라벨을 통합 유무로 변환합니다.
- 활성 질환 라벨이 0(무) → 정상(0)
- 활성 질환 라벨이 1 이상(유·상·하 등) → 비정상(1)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.classifier.dataset import EyeDiseaseDataset, get_transforms

BINARY_NUM_CLASSES = 2
BINARY_LABEL_NAMES = {0: "정상", 1: "비정상"}


def disease_labels_to_binary(label_dict: Dict[str, int]) -> int:
    """질환별 라벨 dict → 통합 이진 라벨."""
    for value in label_dict.values():
        if value >= 0:
            return 1 if value > 0 else 0
    return 0


class BinaryEyeDiseaseDataset(Dataset):
    """EyeDiseaseDataset 래퍼 — 정상(0) / 비정상(1) 단일 라벨."""

    def __init__(self, base: EyeDiseaseDataset):
        self.base = base
        self.animal_type = base.animal_type
        self.samples = base.samples

        counts = self.get_class_counts()
        print(f"\n✓ 이진 데이터셋 (정상 vs 비정상):")
        print(f"  - 정상(0): {counts[0]:,}")
        print(f"  - 비정상(1): {counts[1]:,}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label_dict = self.base[idx]
        binary = disease_labels_to_binary(
            {k: v.item() for k, v in label_dict.items()}
        )
        return image, torch.tensor(binary, dtype=torch.long)

    def get_class_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for _, label_dict in self.samples:
            counts[disease_labels_to_binary(label_dict)] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """클래스 빈도 역수 (Focal Loss alpha / CE weight)."""
        counts = self.get_class_counts()
        weights = torch.tensor(
            [1.0 / (counts[c] + 1e-6) for c in range(BINARY_NUM_CLASSES)],
            dtype=torch.float32,
        )
        return weights / weights.sum() * BINARY_NUM_CLASSES

    def get_sample_weights(self) -> List[float]:
        """WeightedRandomSampler용 샘플 가중치."""
        counts = self.get_class_counts()
        class_w = {c: 1.0 / (counts[c] + 1e-6) for c in range(BINARY_NUM_CLASSES)}
        weights: List[float] = []
        for _, label_dict in self.samples:
            b = disease_labels_to_binary(label_dict)
            weights.append(class_w[b])
        return weights


def create_binary_dataloader(
    data_paths: List[str],
    animal_type: str,
    batch_size: int = 16,
    img_size: int = 300,
    is_training: bool = True,
    num_workers: int = 4,
    use_sampler: bool = True,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """이진 분류 DataLoader."""
    transform = get_transforms(img_size, is_training, aug_preset="train")

    base = EyeDiseaseDataset(
        data_paths=data_paths,
        animal_type=animal_type,
        transform=transform,
        is_training=is_training,
    )
    dataset = BinaryEyeDiseaseDataset(base)

    sampler = None
    shuffle = is_training
    if is_training and use_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
        print(f"✓ WeightedRandomSampler 적용 (이진 클래스 균형)")

    use_pin = pin_memory if pin_memory is not None else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=is_training,
    )

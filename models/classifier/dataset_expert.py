"""
질환 분류 전문가(Expert) 데이터셋.

비정상(질환 있음) 샘플만 사용 — "무"(정상) 제외.
각 샘플을 단일 N-class 질환 라벨로 변환.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.classifier.dataset import EyeDiseaseDataset, get_transforms

# 고양이 5-class (비정상 전문가)
CAT_DISEASE_TO_IDX: Dict[str, int] = {
    "각막궤양": 0,
    "각막부골편": 1,
    "결막염": 2,
    "비궤양성각막염": 3,
    "안검염": 4,
}

# 강아지 10-class (동일 패턴)
DOG_DISEASE_TO_IDX: Dict[str, int] = {
    "결막염": 0,
    "궤양성각막질환": 1,
    "백내장": 2,
    "비궤양성각막질환": 3,
    "색소침착성각막염": 4,
    "안검내반증": 5,
    "안검염": 6,
    "안검종양": 7,
    "유루증": 8,
    "핵경화": 9,
}


def get_disease_mapping(animal_type: str) -> Dict[str, int]:
    animal_type = animal_type.lower()
    if animal_type == "cat":
        return CAT_DISEASE_TO_IDX
    if animal_type == "dog":
        return DOG_DISEASE_TO_IDX
    raise ValueError(f"animal_type은 'dog' 또는 'cat'이어야 합니다: {animal_type}")


def get_active_disease_label(label_dict: Dict[str, int]) -> Tuple[Optional[str], int]:
    """활성 질환명과 라벨 인덱스 반환."""
    for disease, value in label_dict.items():
        if value >= 0:
            return disease, value
    return None, -1


def sample_to_expert_class(
    label_dict: Dict[str, int],
    disease_to_idx: Dict[str, int],
) -> Optional[int]:
    """비정상(라벨>0)이면 질환 class index, 정상이면 None."""
    disease, label = get_active_disease_label(label_dict)
    if disease is None or label <= 0:
        return None
    if disease not in disease_to_idx:
        return None
    return disease_to_idx[disease]


class ExpertEyeDiseaseDataset(Dataset):
    """비정상 샘플만 — 질환 단일 N-class 분류."""

    def __init__(self, base: EyeDiseaseDataset, disease_to_idx: Dict[str, int]):
        self.base = base
        self.animal_type = base.animal_type
        self.disease_to_idx = disease_to_idx
        self.num_classes = len(disease_to_idx)
        self.class_names = [""] * self.num_classes
        for name, idx in disease_to_idx.items():
            self.class_names[idx] = name

        self.indices: List[int] = []
        for i, (_, label_dict) in enumerate(base.samples):
            if sample_to_expert_class(label_dict, disease_to_idx) is not None:
                self.indices.append(i)

        counts = self.get_class_counts()
        print(f"\n✓ Expert 데이터셋 (비정상만, {self.num_classes}-class):")
        print(f"  - 총 샘플: {len(self.indices):,} / 원본 {len(base.samples):,}")
        for c in range(self.num_classes):
            print(f"  - [{c}] {self.class_names[c]}: {counts[c]:,}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.indices[idx]
        image, label_dict = self.base[base_idx]
        label_dict_int = {k: v.item() for k, v in label_dict.items()}
        expert_cls = sample_to_expert_class(label_dict_int, self.disease_to_idx)
        return image, torch.tensor(expert_cls, dtype=torch.long)

    def _expert_labels(self) -> List[int]:
        labels: List[int] = []
        for base_idx in self.indices:
            _, label_dict = self.base.samples[base_idx]
            cls = sample_to_expert_class(label_dict, self.disease_to_idx)
            if cls is not None:
                labels.append(cls)
        return labels

    def get_class_counts(self) -> Dict[int, int]:
        counts = {c: 0 for c in range(self.num_classes)}
        for cls in self._expert_labels():
            counts[cls] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        counts = self.get_class_counts()
        weights = torch.tensor(
            [1.0 / (counts[c] + 1e-6) for c in range(self.num_classes)],
            dtype=torch.float32,
        )
        return weights / weights.sum() * self.num_classes

    def get_sample_weights(self) -> List[float]:
        counts = self.get_class_counts()
        class_w = {c: 1.0 / (counts[c] + 1e-6) for c in range(self.num_classes)}
        weights: List[float] = []
        for base_idx in self.indices:
            _, label_dict = self.base.samples[base_idx]
            cls = sample_to_expert_class(label_dict, self.disease_to_idx)
            weights.append(class_w[cls] if cls is not None else 1.0)
        return weights


def create_expert_dataloader(
    data_paths: List[str],
    animal_type: str,
    batch_size: int = 16,
    img_size: int = 300,
    is_training: bool = True,
    num_workers: int = 4,
    use_sampler: bool = True,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Expert 질환 분류 DataLoader."""
    disease_to_idx = get_disease_mapping(animal_type)
    transform = get_transforms(img_size, is_training, aug_preset="train")

    base = EyeDiseaseDataset(
        data_paths=data_paths,
        animal_type=animal_type,
        transform=transform,
        is_training=is_training,
    )
    dataset = ExpertEyeDiseaseDataset(base, disease_to_idx)

    if len(dataset) == 0:
        raise RuntimeError("Expert 데이터셋이 비어 있습니다. 비정상(유) 샘플 경로를 확인하세요.")

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
        print("✓ WeightedRandomSampler 적용 (질환 클래스 균형)")

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

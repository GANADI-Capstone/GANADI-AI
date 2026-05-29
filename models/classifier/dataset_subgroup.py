"""
부위별 세부 질환 분류 데이터셋.

특정 GROUP_NAME(안검·각막·기타 등) 내 질환만 필터 → subgroup class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.classifier.dataset import EyeDiseaseDataset, get_transforms
from models.classifier.dataset_augment import rand_augment_transforms
from models.classifier.dataset_group import (
    get_active_disease_label,
    get_group_map,
)


def get_subgroup_diseases(animal_type: str, group_name: str) -> List[str]:
    group_map = get_group_map(animal_type)
    if group_name not in group_map:
        raise ValueError(
            f"GROUP_NAME='{group_name}' 없음. 가능: {list(group_map.keys())}"
        )
    return list(group_map[group_name])


def build_disease_to_subgroup(diseases: List[str]) -> Dict[str, int]:
    return {d: i for i, d in enumerate(diseases)}


def sample_to_subgroup_label(
    label_dict: Dict[str, int],
    disease_to_sub: Dict[str, int],
) -> Optional[int]:
    disease, label = get_active_disease_label(label_dict)
    if disease is None or label <= 0:
        return None
    return disease_to_sub.get(disease)


def subgroup_trainable(group_name: str, animal_type: str) -> bool:
    """그룹 내 질환 2개 이상일 때만 세부 모델 학습."""
    return len(get_subgroup_diseases(animal_type, group_name)) >= 2


class SubgroupEyeDiseaseDataset(Dataset):
    """특정 부위 그룹 내 비정상 샘플 → 세부 질환 class."""

    def __init__(self, base: EyeDiseaseDataset, animal_type: str, group_name: str):
        self.base = base
        self.animal_type = animal_type.lower()
        self.group_name = group_name
        self.disease_names = get_subgroup_diseases(animal_type, group_name)
        self.disease_to_sub = build_disease_to_subgroup(self.disease_names)
        self.num_classes = len(self.disease_names)
        self.class_names = self.disease_names

        self.indices: List[int] = []
        for i, (_, label_dict) in enumerate(base.samples):
            if sample_to_subgroup_label(label_dict, self.disease_to_sub) is not None:
                self.indices.append(i)

        counts = self.get_class_counts()
        print(f"\n✓ Subgroup 데이터셋 [{group_name}] ({self.num_classes}-class):")
        print(f"  - 샘플: {len(self.indices):,}")
        for i, name in enumerate(self.disease_names):
            print(f"  - [{i}] {name}: {counts[i]:,}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.indices[idx]
        image, label_dict = self.base[base_idx]
        label_dict_int = {k: v.item() for k, v in label_dict.items()}
        sub_cls = sample_to_subgroup_label(label_dict_int, self.disease_to_sub)
        return image, torch.tensor(sub_cls, dtype=torch.long)

    def _sub_labels(self) -> List[int]:
        out: List[int] = []
        for base_idx in self.indices:
            _, label_dict = self.base.samples[base_idx]
            cls = sample_to_subgroup_label(label_dict, self.disease_to_sub)
            if cls is not None:
                out.append(cls)
        return out

    def get_class_counts(self) -> Dict[int, int]:
        counts = {c: 0 for c in range(self.num_classes)}
        for cls in self._sub_labels():
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
            cls = sample_to_subgroup_label(label_dict, self.disease_to_sub)
            weights.append(class_w[cls] if cls is not None else 1.0)
        return weights


def create_subgroup_dataloader(
    data_paths: List[str],
    animal_type: str,
    group_name: str,
    batch_size: int = 16,
    img_size: int = 300,
    is_training: bool = True,
    num_workers: int = 4,
    use_sampler: bool = True,
    pin_memory: Optional[bool] = None,
    aug_preset: str = "train",
) -> DataLoader:
    if not subgroup_trainable(group_name, animal_type):
        diseases = get_subgroup_diseases(animal_type, group_name)
        raise RuntimeError(
            f"GROUP_NAME='{group_name}' 질환 {len(diseases)}개 — "
            "세부 모델 불필요 (2개 이상일 때만 학습)."
        )

    if is_training and aug_preset == "rand":
        transform = rand_augment_transforms(img_size, num_ops=2, magnitude=9)
        print(f"✓ RandAugment (num_ops=2, magnitude=9, {img_size}px)")
    else:
        transform = get_transforms(img_size, is_training, aug_preset=aug_preset)
    base = EyeDiseaseDataset(
        data_paths=data_paths,
        animal_type=animal_type,
        transform=transform,
        is_training=is_training,
    )
    dataset = SubgroupEyeDiseaseDataset(base, animal_type, group_name)

    if len(dataset) == 0:
        raise RuntimeError(f"Subgroup [{group_name}] 데이터셋이 비어 있습니다.")

    sampler = None
    shuffle = is_training
    if is_training and use_sampler:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print("✓ WeightedRandomSampler (세부 질환 균형)")

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

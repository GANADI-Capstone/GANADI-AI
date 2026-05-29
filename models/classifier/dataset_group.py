"""
강아지·고양이 부위 그룹 분류 데이터셋.

비정상(질환 있음) 샘플만 사용 → 질환명을 부위 그룹 라벨로 매핑.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.classifier.dataset import EyeDiseaseDataset, get_transforms

# 강아지 10질환 → 5 부위 그룹
DOG_GROUP_MAP: Dict[str, List[str]] = {
    "안검": ["안검내반증", "안검염", "안검종양"],
    "결막": ["결막염"],
    "각막": ["궤양성각막질환", "비궤양성각막질환", "색소침착성각막염"],
    "수정체": ["백내장"],
    "기타": ["유루증", "핵경화"],
}

DOG_GROUP_LABEL: Dict[str, int] = {
    "안검": 0,
    "결막": 1,
    "각막": 2,
    "수정체": 3,
    "기타": 4,
}

# 고양이 (그룹 분류 의미는 약하지만 동일 패턴 지원)
CAT_GROUP_MAP: Dict[str, List[str]] = {
    "안검": ["안검염"],
    "결막": ["결막염"],
    "각막": ["각막궤양", "비궤양성각막염", "각막부골편"],
}

CAT_GROUP_LABEL: Dict[str, int] = {
    "안검": 0,
    "결막": 1,
    "각막": 2,
}

# 체크포인트 파일명 slug (subgroup용)
GROUP_SLUG: Dict[str, str] = {
    "안검": "eyelid",
    "결막": "conjunctiva",
    "각막": "cornea",
    "수정체": "lens",
    "기타": "etc",
}


def get_group_map(animal_type: str) -> Dict[str, List[str]]:
    if animal_type.lower() == "dog":
        return DOG_GROUP_MAP
    if animal_type.lower() == "cat":
        return CAT_GROUP_MAP
    raise ValueError(f"animal_type은 'dog' 또는 'cat': {animal_type}")


def get_group_label_map(animal_type: str) -> Dict[str, int]:
    if animal_type.lower() == "dog":
        return DOG_GROUP_LABEL
    if animal_type.lower() == "cat":
        return CAT_GROUP_LABEL
    raise ValueError(f"animal_type은 'dog' 또는 'cat': {animal_type}")


def build_disease_to_group(animal_type: str) -> Dict[str, int]:
    group_map = get_group_map(animal_type)
    label_map = get_group_label_map(animal_type)
    mapping: Dict[str, int] = {}
    for group_name, diseases in group_map.items():
        idx = label_map[group_name]
        for disease in diseases:
            mapping[disease] = idx
    return mapping


def get_active_disease_label(label_dict: Dict[str, int]) -> Tuple[Optional[str], int]:
    for disease, value in label_dict.items():
        if value >= 0:
            return disease, value
    return None, -1


def sample_to_group_label(
    label_dict: Dict[str, int],
    disease_to_group: Dict[str, int],
) -> Optional[int]:
    disease, label = get_active_disease_label(label_dict)
    if disease is None or label <= 0:
        return None
    return disease_to_group.get(disease)


class GroupEyeDiseaseDataset(Dataset):
    """비정상 샘플 → 부위 그룹 N-class."""

    def __init__(self, base: EyeDiseaseDataset, animal_type: str):
        self.base = base
        self.animal_type = animal_type.lower()
        self.group_label_map = get_group_label_map(self.animal_type)
        self.disease_to_group = build_disease_to_group(self.animal_type)
        self.num_classes = len(self.group_label_map)

        self.class_names = [""] * self.num_classes
        for group_name, idx in self.group_label_map.items():
            self.class_names[idx] = group_name

        self.indices: List[int] = []
        for i, (_, label_dict) in enumerate(base.samples):
            if sample_to_group_label(label_dict, self.disease_to_group) is not None:
                self.indices.append(i)

        counts = self.get_class_counts()
        print(f"\n✓ Group 데이터셋 ({self.animal_type}, {self.num_classes}-class):")
        print(f"  - 샘플: {len(self.indices):,} / {len(base.samples):,}")
        for c in range(self.num_classes):
            print(f"  - [{c}] {self.class_names[c]}: {counts[c]:,}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.indices[idx]
        image, label_dict = self.base[base_idx]
        label_dict_int = {k: v.item() for k, v in label_dict.items()}
        group_cls = sample_to_group_label(label_dict_int, self.disease_to_group)
        return image, torch.tensor(group_cls, dtype=torch.long)

    def _group_labels(self) -> List[int]:
        out: List[int] = []
        for base_idx in self.indices:
            _, label_dict = self.base.samples[base_idx]
            cls = sample_to_group_label(label_dict, self.disease_to_group)
            if cls is not None:
                out.append(cls)
        return out

    def get_class_counts(self) -> Dict[int, int]:
        counts = {c: 0 for c in range(self.num_classes)}
        for cls in self._group_labels():
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
            cls = sample_to_group_label(label_dict, self.disease_to_group)
            weights.append(class_w[cls] if cls is not None else 1.0)
        return weights


def create_group_dataloader(
    data_paths: List[str],
    animal_type: str,
    batch_size: int = 16,
    img_size: int = 300,
    is_training: bool = True,
    num_workers: int = 4,
    use_sampler: bool = True,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    transform = get_transforms(img_size, is_training, aug_preset="train")
    base = EyeDiseaseDataset(
        data_paths=data_paths,
        animal_type=animal_type,
        transform=transform,
        is_training=is_training,
    )
    dataset = GroupEyeDiseaseDataset(base, animal_type)

    if len(dataset) == 0:
        raise RuntimeError("Group 데이터셋이 비어 있습니다.")

    sampler = None
    shuffle = is_training
    if is_training and use_sampler:
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print("✓ WeightedRandomSampler (부위 그룹 균형)")

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

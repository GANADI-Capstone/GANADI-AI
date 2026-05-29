"""
TL(학습용 라벨링) 데이터만 사용하는 random train/val 분할.

VL 폴더는 device 편향이 있어 사용하지 않습니다.

환경변수:
  SPLIT_SEED=42
  VAL_RATIO=0.2
  USE_GROUP_SPLIT=1  — crop_D* 그룹 단위 분할 (기본 ON)
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.classifier.dataset import EyeDiseaseDataset, get_transforms

try:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

GROUP_RE = re.compile(r"crop_(D\d+)_", re.IGNORECASE)

DOG_TL_PATHS = [
    "eye_data/개/안구/일반",
    "eye_data/TL2/개/안구/일반",
]
CAT_TL_PATHS = [
    "eye_data/TL2/고양이/안구/일반",
]

SMARTPHONE = "스마트폰"
MEDICAL_DEVICES = frozenset({"검안경", "일반카메라"})


@dataclass
class SampleMeta:
    index: int
    img_path: str
    disease: str
    label: int
    stratum: str
    group_id: str
    device: str


def get_tl_paths(animal_type: str) -> List[str]:
    if animal_type.lower() == "dog":
        return list(DOG_TL_PATHS)
    if animal_type.lower() == "cat":
        return list(CAT_TL_PATHS)
    raise ValueError(f"animal_type은 'dog' 또는 'cat': {animal_type}")


def extract_group_id(img_path: str) -> str:
    """파일명 crop_D6_... → D6 (동일 촬영 세션 근사)."""
    base = os.path.basename(img_path)
    m = GROUP_RE.search(base)
    if m:
        return m.group(1).upper()
    stem = os.path.splitext(base)[0]
    return stem[:32]


def load_device_from_image_path(img_path: str) -> str:
    json_path = os.path.splitext(img_path)[0] + ".json"
    if not os.path.isfile(json_path):
        return "(missing_json)"
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        imgs = data.get("images")
        if isinstance(imgs, dict):
            return imgs.get("meta", {}).get("device", "") or "(empty)"
        if isinstance(imgs, list) and imgs:
            return imgs[0].get("meta", {}).get("device", "") or "(empty)"
    except Exception:
        pass
    return "(read_error)"


def build_sample_metadata(base: EyeDiseaseDataset) -> List[SampleMeta]:
    metas: List[SampleMeta] = []
    for idx, (img_path, label_dict) in enumerate(base.samples):
        disease = next(d for d, v in label_dict.items() if v >= 0)
        label = label_dict[disease]
        metas.append(
            SampleMeta(
                index=idx,
                img_path=img_path,
                disease=disease,
                label=label,
                stratum=f"{disease}_{label}",
                group_id=extract_group_id(img_path),
                device=load_device_from_image_path(img_path),
            )
        )
    return metas


def _split_indices_group_global(
    metas: Sequence[SampleMeta],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """crop_D* 그룹 전역 할당 — 동일 D코드는 train 또는 val 한쪽만."""
    rng = random.Random(seed)
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for m in metas:
        group_to_indices[m.group_id].append(m.index)

    group_ids = list(group_to_indices.keys())
    rng.shuffle(group_ids)
    n_val = max(1, int(round(len(group_ids) * val_ratio)))
    n_val = min(n_val, len(group_ids) - 1) if len(group_ids) > 1 else 1
    val_groups = set(group_ids[:n_val])

    train_idx: List[int] = []
    val_idx: List[int] = []
    for gid, idxs in group_to_indices.items():
        if gid in val_groups:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _split_indices_group_stratified(
    metas: Sequence[SampleMeta],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """질환×라벨 stratum 내 crop_D* 그룹 분할 (stratum 균형 우선)."""
    rng = random.Random(seed)
    by_stratum: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for m in metas:
        by_stratum[m.stratum][m.group_id].append(m.index)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for stratum, groups in sorted(by_stratum.items()):
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        if len(group_ids) == 1:
            val_groups = {group_ids[0]}
        else:
            n_val = max(1, int(round(len(group_ids) * val_ratio)))
            n_val = min(n_val, len(group_ids) - 1)
            val_groups = set(group_ids[:n_val])

        for gid, idxs in groups.items():
            if gid in val_groups:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _split_indices_sklearn_group(
    metas: Sequence[SampleMeta],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    indices = [m.index for m in metas]
    groups = [m.group_id for m in metas]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_pos, val_pos = next(gss.split(indices, groups=groups))
    return [indices[i] for i in train_pos], [indices[i] for i in val_pos]


def _split_indices_sklearn_stratify(
    metas: Sequence[SampleMeta],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    indices = [m.index for m in metas]
    strata = [m.stratum for m in metas]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=strata,
    )
    return list(train_idx), list(val_idx)


def split_tl_indices(
    base: EyeDiseaseDataset,
    *,
    val_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    use_group_split: Optional[bool] = None,
) -> Tuple[List[int], List[int], List[SampleMeta], Dict[str, object]]:
    val_ratio = float(os.environ.get("VAL_RATIO", "0.2") if val_ratio is None else val_ratio)
    seed = int(os.environ.get("SPLIT_SEED", "42") if seed is None else seed)
    split_mode = os.environ.get("GROUP_SPLIT_MODE", "global").strip().lower()
    if use_group_split is None:
        use_group_split = os.environ.get("USE_GROUP_SPLIT", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    metas = build_sample_metadata(base)
    method = "group_global"

    if use_group_split:
        if split_mode in ("stratified", "per_stratum"):
            method = "group_stratified"
            train_idx, val_idx = _split_indices_group_stratified(metas, val_ratio, seed)
        else:
            train_idx, val_idx = _split_indices_group_global(metas, val_ratio, seed)
    elif HAS_SKLEARN:
        method = "sklearn_group_shuffle"
        train_idx, val_idx = _split_indices_sklearn_group(metas, val_ratio, seed)
    else:
        method = "group_stratified_fallback"
        train_idx, val_idx = _split_indices_group_stratified(metas, val_ratio, seed)

    meta = {
        "method": method,
        "seed": seed,
        "val_ratio": val_ratio,
        "use_group_split": use_group_split,
        "group_split_mode": split_mode if use_group_split else None,
        "has_sklearn": HAS_SKLEARN,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
    }
    return train_idx, val_idx, metas, meta


def _device_binary_label(meta: SampleMeta) -> int:
    """무(정상)=0, 유/상/하 등=1 — device×라벨 교차 확인용."""
    return 0 if meta.label == 0 else 1


def print_split_distribution(
    metas: Sequence[SampleMeta],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    title: str = "Random Split 분포",
) -> Dict[str, object]:
    idx_set_train = set(train_idx)
    idx_set_val = set(val_idx)
    train_m = [m for m in metas if m.index in idx_set_train]
    val_m = [m for m in metas if m.index in idx_set_val]

    print(f"\n{'=' * 64}")
    print(f"📊 {title}")
    print(f"{'=' * 64}")
    print(f"  Train: {len(train_m):,}  Val: {len(val_m):,}  Total: {len(metas):,}")

    def _print_device_block(name: str, subset: Sequence[SampleMeta]) -> Dict[str, object]:
        print(f"\n  [{name}] device × 이진라벨")
        for binary, bname in [(0, "정상(무)"), (1, "비정상")]:
            sub = [m for m in subset if _device_binary_label(m) == binary]
            c = Counter(m.device for m in sub)
            print(f"    {bname} n={len(sub):,}: {dict(c)}")

        sp_abn = sum(
            1 for m in subset if m.device == SMARTPHONE and _device_binary_label(m) == 1
        )
        med_norm = sum(
            1
            for m in subset
            if m.device in MEDICAL_DEVICES and _device_binary_label(m) == 0
        )
        print(f"    ★ 스마트폰+비정상: {sp_abn:,}")
        print(f"    ★ 의료장비+정상(무): {med_norm:,}")

        strata = Counter(m.stratum for m in subset)
        print(f"    stratum(질환_라벨) 종류: {len(strata)}")
        return {
            "smartphone_abnormal": sp_abn,
            "medical_normal": med_norm,
            "device_normal": dict(Counter(m.device for m in subset if _device_binary_label(m) == 0)),
            "device_abnormal": dict(Counter(m.device for m in subset if _device_binary_label(m) == 1)),
        }

    train_stats = _print_device_block("Train", train_m)
    val_stats = _print_device_block("Val", val_m)

    overlap_groups = set(m.group_id for m in train_m) & set(m.group_id for m in val_m)
    print(f"\n  crop_D* group train∩val overlap: {len(overlap_groups)} (0이어야 함)")

    return {
        "train_size": len(train_m),
        "val_size": len(val_m),
        "train": train_stats,
        "val": val_stats,
        "group_overlap": len(overlap_groups),
    }


class RandomSplitEyeDataset(Dataset):
    """EyeDiseaseDataset의 index subset + device 메타."""

    def __init__(
        self,
        base: EyeDiseaseDataset,
        indices: List[int],
        device_by_index: Dict[int, str],
    ):
        self.base = base
        self.indices = indices
        self.device_by_index = device_by_index
        self.animal_type = base.animal_type
        self.label_map = base.label_map
        self.diseases = base.diseases

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]

    def get_device(self, i: int) -> str:
        return self.device_by_index.get(self.indices[i], "(unknown)")

    def get_class_weights(self, disease: str) -> torch.Tensor:
        labels = []
        for idx in self.indices:
            _, label_dict = self.base.samples[idx]
            if label_dict[disease] >= 0:
                labels.append(label_dict[disease])
        if not labels:
            return torch.ones(len(self.label_map[disease]))
        num_classes = len(self.label_map[disease])
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        weights = 1.0 / (counts.float() + 1e-6)
        return weights / weights.sum() * num_classes

    def get_sample_weights(
        self,
        boost_disease: Optional[str] = None,
        boost_minority_factor: float = 2.0,
    ) -> List[float]:
        disease_class_counts: Dict[str, Dict[int, int]] = {}
        for disease in self.diseases:
            cc: Dict[int, int] = defaultdict(int)
            for idx in self.indices:
                _, label_dict = self.base.samples[idx]
                lv = label_dict[disease]
                if lv >= 0:
                    cc[lv] += 1
            disease_class_counts[disease] = dict(cc)

        minority_label: Optional[int] = None
        if boost_disease and boost_disease in self.diseases:
            counts = disease_class_counts.get(boost_disease, {})
            if len(counts) >= 2:
                minority_label = min(counts.keys(), key=lambda k: counts[k])

        weights: List[float] = []
        for idx in self.indices:
            _, label_dict = self.base.samples[idx]
            per_dis = []
            for disease in self.diseases:
                lv = label_dict[disease]
                if lv >= 0:
                    cnt = disease_class_counts[disease].get(lv, 1)
                    per_dis.append(1.0 / cnt)
            w = sum(per_dis) / len(per_dis) if per_dis else 1.0
            if (
                minority_label is not None
                and boost_disease
                and label_dict.get(boost_disease, -1) == minority_label
            ):
                w *= boost_minority_factor
            weights.append(w)
        return weights


def create_random_split_dataloaders(
    animal_type: str,
    batch_size: int = 16,
    img_size: int = 300,
    num_workers: int = 4,
    use_sampler: bool = False,
    sampler_boost_disease: Optional[str] = None,
    sampler_boost_factor: float = 2.0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, RandomSplitEyeDataset, RandomSplitEyeDataset, Dict[str, object]]:
    """TL 전체 수집 → random split → train/val DataLoader."""
    tl_paths = get_tl_paths(animal_type)

    print("\n📁 TL 데이터 수집 (VL 미사용)...")
    base_train = EyeDiseaseDataset(
        data_paths=tl_paths,
        animal_type=animal_type,
        transform=get_transforms(img_size, is_training=True, aug_preset="train"),
        is_training=True,
    )
    base_eval = EyeDiseaseDataset(
        data_paths=tl_paths,
        animal_type=animal_type,
        transform=get_transforms(img_size, is_training=False),
        is_training=False,
    )

    train_idx, val_idx, metas, split_meta = split_tl_indices(base_train)
    device_map = {m.index: m.device for m in metas}
    dist = print_split_distribution(metas, train_idx, val_idx)

    train_ds = RandomSplitEyeDataset(base_train, train_idx, device_map)
    val_ds = RandomSplitEyeDataset(base_eval, val_idx, device_map)

    sampler = None
    shuffle = True
    if use_sampler:
        weights = train_ds.get_sample_weights(
            boost_disease=sampler_boost_disease,
            boost_minority_factor=sampler_boost_factor,
        )
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        print("✓ WeightedRandomSampler (train split)")

    use_pin = pin_memory if pin_memory is not None else False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=False,
    )

    meta = {**split_meta, "distribution": dist, "tl_paths": tl_paths}
    return train_loader, val_loader, train_ds, val_ds, meta


if __name__ == "__main__":
    import sys

    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    for animal in ("dog", "cat"):
        print("\n" + "#" * 64)
        print(f"# ANIMAL_TYPE={animal}")
        print("#" * 64)
        _, _, _, _, meta = create_random_split_dataloaders(
            animal, batch_size=4, num_workers=0, use_sampler=False
        )
        print(f"\n→ {meta['train_size']:,} train / {meta['val_size']:,} val")

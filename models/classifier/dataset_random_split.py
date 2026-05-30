"""
TL(학습용 라벨링) 데이터만 사용하는 random train/val 분할.

VL 폴더는 device 편향이 있어 사용하지 않습니다.

환경변수:
  MODE=multitask|group|subgroup  (기본 multitask)
  GROUP_NAME=안검             — MODE=subgroup 일 때 필수
  MAX_PER_CLASS=5000            — 클래스당 상한 (group/subgroup, 0=무제한)
  DISEASE_CAPS='{"핵경화":7000,...}' — 질환별 cap (미명시 질환은 MAX_PER_CLASS/stratum)
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
from models.classifier.dataset_group import (
    build_disease_to_group,
    get_group_label_map,
    sample_to_group_label,
)
from models.classifier.dataset_subgroup import (
    build_disease_to_subgroup,
    get_subgroup_diseases,
    sample_to_subgroup_label,
)

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


SPLIT_MODES = frozenset({"multitask", "group", "subgroup"})


@dataclass
class SampleMeta:
    index: int
    img_path: str
    disease: str
    label: int
    stratum: str
    group_id: str
    device: str
    task_class: Optional[int] = None


def resolve_split_mode(mode: Optional[str] = None) -> str:
    raw = (mode or os.environ.get("MODE", "multitask")).strip().lower()
    if raw not in SPLIT_MODES:
        raise ValueError(f"MODE는 {sorted(SPLIT_MODES)} 중 하나: {raw}")
    return raw


def resolve_max_per_class(max_per_class: Optional[int] = None) -> Optional[int]:
    if max_per_class is not None:
        return max_per_class if max_per_class > 0 else None
    raw = os.environ.get("MAX_PER_CLASS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def resolve_disease_caps(disease_caps: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """DISEASE_CAPS JSON — 질환명 → cap (질환 전체 strata 합산)."""
    if disease_caps is not None:
        return {k: int(v) for k, v in disease_caps.items() if int(v) > 0}
    raw = os.environ.get("DISEASE_CAPS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"DISEASE_CAPS JSON 파싱 실패: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("DISEASE_CAPS는 JSON 객체여야 합니다.")
    return {str(k): int(v) for k, v in data.items() if int(v) > 0}


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
                task_class=None,
            )
        )
    return metas


def build_group_sample_metadata(
    base: EyeDiseaseDataset,
    animal_type: str,
) -> List[SampleMeta]:
    """비정상 샘플 → 부위 그룹 class (MODE=group)."""
    disease_to_group = build_disease_to_group(animal_type)
    metas: List[SampleMeta] = []
    for idx, (img_path, label_dict) in enumerate(base.samples):
        group_cls = sample_to_group_label(label_dict, disease_to_group)
        if group_cls is None:
            continue
        disease = next(d for d, v in label_dict.items() if v >= 0)
        metas.append(
            SampleMeta(
                index=idx,
                img_path=img_path,
                disease=disease,
                label=label_dict[disease],
                stratum=f"group_{group_cls}",
                group_id=extract_group_id(img_path),
                device=load_device_from_image_path(img_path),
                task_class=group_cls,
            )
        )
    return metas


def build_subgroup_sample_metadata(
    base: EyeDiseaseDataset,
    animal_type: str,
    group_name: str,
) -> List[SampleMeta]:
    """특정 부위 그룹 내 비정상 → 세부 질환 class (MODE=subgroup)."""
    diseases = get_subgroup_diseases(animal_type, group_name)
    disease_to_sub = build_disease_to_subgroup(diseases)
    metas: List[SampleMeta] = []
    for idx, (img_path, label_dict) in enumerate(base.samples):
        sub_cls = sample_to_subgroup_label(label_dict, disease_to_sub)
        if sub_cls is None:
            continue
        disease = diseases[sub_cls]
        metas.append(
            SampleMeta(
                index=idx,
                img_path=img_path,
                disease=disease,
                label=label_dict[disease],
                stratum=f"sub_{disease}",
                group_id=extract_group_id(img_path),
                device=load_device_from_image_path(img_path),
                task_class=sub_cls,
            )
        )
    return metas


def apply_max_per_class_cap(
    metas: Sequence[SampleMeta],
    max_per_class: int,
    seed: int,
) -> Tuple[List[SampleMeta], Dict[str, object]]:
    """stratum(클래스)당 max_per_class 초과분 랜덤 서브샘플."""
    return apply_sample_caps(metas, max_per_class=max_per_class, disease_caps=None, seed=seed)


def apply_sample_caps(
    metas: Sequence[SampleMeta],
    *,
    max_per_class: Optional[int] = None,
    disease_caps: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Tuple[List[SampleMeta], Dict[str, object]]:
    """샘플 cap — DISEASE_CAPS 질환은 질환 단위, 나머지는 stratum×MAX_PER_CLASS."""
    disease_caps = disease_caps or {}
    if not max_per_class and not disease_caps:
        return list(metas), {}

    rng = random.Random(seed)
    groups: Dict[str, List[SampleMeta]] = defaultdict(list)
    limits: Dict[str, int] = {}

    for m in metas:
        if m.disease in disease_caps:
            key = f"disease:{m.disease}"
            limits[key] = disease_caps[m.disease]
        elif max_per_class:
            key = f"stratum:{m.stratum}"
            limits[key] = max_per_class
        else:
            key = f"pass:{m.index}"
            limits[key] = 10**12
        groups[key].append(m)

    capped: List[SampleMeta] = []
    stats: Dict[str, object] = {}
    for key, items in sorted(groups.items()):
        before = len(items)
        limit = limits[key]
        if before > limit:
            rng.shuffle(items)
            items = items[:limit]
        capped.extend(items)
        stats[key] = {"before": before, "after": len(items), "limit": limit}

    rng.shuffle(capped)
    print(f"\n✂️  Sample cap (seed={seed})")
    if max_per_class:
        print(f"  MAX_PER_CLASS (stratum): {max_per_class:,}")
    if disease_caps:
        print(f"  DISEASE_CAPS: {disease_caps}")
    for key, st in stats.items():
        b, a, lim = st["before"], st["after"], st["limit"]
        note = f" → {a:,}" if b != a else ""
        print(f"  [{key}] limit={lim:,}: {b:,}{note}")
    print(f"  Total: {len(metas):,} → {len(capped):,}")
    return capped, stats


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


def split_indices_from_metas(
    metas: Sequence[SampleMeta],
    *,
    val_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    use_group_split: Optional[bool] = None,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    """SampleMeta 목록 → base index train/val (crop_D* 그룹 분할)."""
    val_ratio = float(os.environ.get("VAL_RATIO", "0.2") if val_ratio is None else val_ratio)
    seed = int(os.environ.get("SPLIT_SEED", "42") if seed is None else seed)
    split_mode = os.environ.get("GROUP_SPLIT_MODE", "global").strip().lower()
    if use_group_split is None:
        use_group_split = os.environ.get("USE_GROUP_SPLIT", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    if not metas:
        raise RuntimeError("split 대상 SampleMeta가 비어 있습니다.")

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
    return train_idx, val_idx, meta


def split_tl_indices(
    base: EyeDiseaseDataset,
    *,
    val_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    use_group_split: Optional[bool] = None,
) -> Tuple[List[int], List[int], List[SampleMeta], Dict[str, object]]:
    metas = build_sample_metadata(base)
    train_idx, val_idx, meta = split_indices_from_metas(
        metas, val_ratio=val_ratio, seed=seed, use_group_split=use_group_split,
    )
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


class RandomSplitTaskDataset(Dataset):
    """단일 class 라벨 (group/subgroup) + device 메타."""

    def __init__(
        self,
        base: EyeDiseaseDataset,
        indices: List[int],
        device_by_index: Dict[int, str],
        task_class_by_index: Dict[int, int],
        class_names: List[str],
    ):
        self.base = base
        self.indices = indices
        self.device_by_index = device_by_index
        self.task_class_by_index = task_class_by_index
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.animal_type = base.animal_type

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.indices[i]
        image, _ = self.base[base_idx]
        cls = self.task_class_by_index[base_idx]
        return image, torch.tensor(cls, dtype=torch.long)

    def get_device(self, i: int) -> str:
        return self.device_by_index.get(self.indices[i], "(unknown)")

    def get_class_counts(self) -> Dict[int, int]:
        counts = {c: 0 for c in range(self.num_classes)}
        for idx in self.indices:
            cls = self.task_class_by_index[idx]
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
        return [class_w[self.task_class_by_index[idx]] for idx in self.indices]


def _build_metas_for_mode(
    base: EyeDiseaseDataset,
    mode: str,
    animal_type: str,
    group_name: Optional[str],
) -> List[SampleMeta]:
    if mode == "multitask":
        return build_sample_metadata(base)
    if mode == "group":
        return build_group_sample_metadata(base, animal_type)
    if mode == "subgroup":
        if not group_name:
            raise ValueError("MODE=subgroup 일 때 GROUP_NAME 환경변수가 필요합니다.")
        return build_subgroup_sample_metadata(base, animal_type, group_name)
    raise ValueError(f"unknown mode: {mode}")


def _task_class_names(mode: str, animal_type: str, group_name: Optional[str]) -> List[str]:
    if mode == "group":
        label_map = get_group_label_map(animal_type)
        names = [""] * len(label_map)
        for name, idx in label_map.items():
            names[idx] = name
        return names
    if mode == "subgroup":
        return get_subgroup_diseases(animal_type, group_name or "")
    return []


def create_random_split_dataloaders(
    animal_type: str,
    batch_size: int = 16,
    img_size: int = 300,
    num_workers: int = 4,
    use_sampler: bool = False,
    sampler_boost_disease: Optional[str] = None,
    sampler_boost_factor: float = 2.0,
    pin_memory: Optional[bool] = None,
    *,
    mode: Optional[str] = None,
    group_name: Optional[str] = None,
    max_per_class: Optional[int] = None,
    disease_caps: Optional[Dict[str, int]] = None,
    aug_preset: str = "train",
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset, Dict[str, object]]:
    """TL 수집 → (필터/cap) → random split → train/val DataLoader.

    MODE:
      multitask — 멀티태스크 질환 (기본, RandomSplitEyeDataset)
      group     — 5-class 부위 (RandomSplitTaskDataset)
      subgroup  — GROUP_NAME 세부 질환 (RandomSplitTaskDataset)
    """
    split_mode = resolve_split_mode(mode)
    cap = resolve_max_per_class(max_per_class)
    d_caps = resolve_disease_caps(disease_caps)
    seed = int(os.environ.get("SPLIT_SEED", "42"))
    group_name = (group_name or os.environ.get("GROUP_NAME", "")).strip() or None
    tl_paths = get_tl_paths(animal_type)

    print(f"\n📁 TL 데이터 수집 (VL 미사용) | MODE={split_mode}")
    if split_mode == "subgroup":
        print(f"  GROUP_NAME={group_name}")
    if cap:
        print(f"  MAX_PER_CLASS={cap:,}")
    if d_caps:
        print(f"  DISEASE_CAPS={d_caps}")

    if aug_preset == "rand":
        from models.classifier.dataset_augment import rand_augment_transforms

        train_transform = rand_augment_transforms(img_size, num_ops=2, magnitude=9)
        print(f"✓ RandAugment (num_ops=2, magnitude=9, {img_size}px)")
    else:
        train_transform = get_transforms(img_size, is_training=True, aug_preset=aug_preset)

    base_train = EyeDiseaseDataset(
        data_paths=tl_paths,
        animal_type=animal_type,
        transform=train_transform,
        is_training=True,
    )
    base_eval = EyeDiseaseDataset(
        data_paths=tl_paths,
        animal_type=animal_type,
        transform=get_transforms(img_size, is_training=False),
        is_training=False,
    )

    metas = _build_metas_for_mode(base_train, split_mode, animal_type, group_name)
    if not metas:
        raise RuntimeError(
            f"MODE={split_mode} 필터 후 샘플 0건. 경로·GROUP_NAME 확인."
        )

    cap_stats: Optional[Dict[str, object]] = None
    if cap is not None or d_caps:
        metas, cap_stats = apply_sample_caps(
            metas, max_per_class=cap, disease_caps=d_caps, seed=seed,
        )

    train_idx, val_idx, split_meta = split_indices_from_metas(metas)
    device_map = {m.index: m.device for m in metas}
    task_class_map = {m.index: m.task_class for m in metas if m.task_class is not None}

    title = f"Random Split ({split_mode})"
    if group_name:
        title += f" [{group_name}]"
    dist = print_split_distribution(metas, train_idx, val_idx, title=title)

    if split_mode == "multitask":
        train_ds: Dataset = RandomSplitEyeDataset(base_train, train_idx, device_map)
        val_ds: Dataset = RandomSplitEyeDataset(base_eval, val_idx, device_map)
        if use_sampler:
            weights = train_ds.get_sample_weights(
                boost_disease=sampler_boost_disease,
                boost_minority_factor=sampler_boost_factor,
            )
        else:
            weights = None
    else:
        class_names = _task_class_names(split_mode, animal_type, group_name)
        train_ds = RandomSplitTaskDataset(
            base_train, train_idx, device_map, task_class_map, class_names,
        )
        val_ds = RandomSplitTaskDataset(
            base_eval, val_idx, device_map, task_class_map, class_names,
        )
        counts = train_ds.get_class_counts()
        print(f"\n✓ Task 데이터셋 ({split_mode}, {len(class_names)}-class):")
        print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")
        for c, name in enumerate(class_names):
            print(f"  - [{c}] {name}: train={counts.get(c, 0):,}")
        weights = train_ds.get_sample_weights() if use_sampler else None

    sampler = None
    shuffle = True
    if use_sampler and weights is not None:
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

    meta = {
        **split_meta,
        "mode": split_mode,
        "group_name": group_name,
        "max_per_class": cap,
        "disease_caps": d_caps or None,
        "cap_stats": cap_stats,
        "distribution": dist,
        "tl_paths": tl_paths,
    }
    if split_mode != "multitask":
        meta["class_names"] = _task_class_names(split_mode, animal_type, group_name)
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

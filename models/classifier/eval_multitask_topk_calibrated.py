"""
멀티태스크 모델 Per-head Temperature Scaling + Top-K 평가.

TL Random Val을 Calibration(50%) / Test(50%)로 나누어
각 질환 헤드별 T를 NLL 최소화로 학습(후처리만)하고 Top-K 성능을 비교합니다.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog python models/classifier/eval_multitask_topk_calibrated.py
  ANIMAL_TYPE=cat python models/classifier/eval_multitask_topk_calibrated.py

환경변수:
  ANIMAL_TYPE=dog|cat
  SPLIT_SEED=42          — TL train/val split (train_random_split과 동일)
  CAL_SPLIT_SEED=43      — Val 내 cal/test 50:50 split
  CAL_RATIO=0.5
  CHECKPOINT=
  BATCH_SIZE=32
  IMG_SIZE=300
  DISEASE_WEIGHTS='{"백내장":0.8}'  — (선택) Top-K 재정렬 가중
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.dataset_random_split import (
    MEDICAL_DEVICES,
    SMARTPHONE,
    RandomSplitEyeDataset,
    create_random_split_dataloaders,
    extract_group_id,
)
from models.classifier.eval_multitask_topk import (
    TOP_KS,
    resolve_checkpoint,
    resolve_disease_weights,
)
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers
from models.classifier.train_random_split import RandomSplitConfig

SampleRecord = Dict[str, object]


def split_val_cal_test_indices(
    val_ds: RandomSplitEyeDataset,
    *,
    cal_ratio: float = 0.5,
    seed: int = 43,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    """Val base indices → crop_D* 그룹 단위 cal/test (누수 방지)."""
    rng = random.Random(seed)
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in val_ds.indices:
        img_path, _ = val_ds.base.samples[idx]
        group_to_indices[extract_group_id(img_path)].append(idx)

    group_ids = list(group_to_indices.keys())
    rng.shuffle(group_ids)
    n_cal = max(1, int(round(len(group_ids) * cal_ratio)))
    n_cal = min(n_cal, len(group_ids) - 1) if len(group_ids) > 1 else 1
    cal_groups = set(group_ids[:n_cal])

    cal_idx: List[int] = []
    test_idx: List[int] = []
    for gid, idxs in group_to_indices.items():
        if gid in cal_groups:
            cal_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    rng.shuffle(cal_idx)
    rng.shuffle(test_idx)
    overlap = set(extract_group_id(val_ds.base.samples[i][0]) for i in cal_idx) & set(
        extract_group_id(val_ds.base.samples[i][0]) for i in test_idx
    )
    meta = {
        "cal_ratio": cal_ratio,
        "cal_split_seed": seed,
        "cal_size": len(cal_idx),
        "test_size": len(test_idx),
        "group_overlap": len(overlap),
    }
    return cal_idx, test_idx, meta


def make_subset_loader(
    val_ds: RandomSplitEyeDataset,
    subset_indices: List[int],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, RandomSplitEyeDataset]:
    sub_ds = RandomSplitEyeDataset(
        val_ds.base, subset_indices, val_ds.device_by_index,
    )
    loader = DataLoader(
        sub_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader, sub_ds


def _active_disease_and_label(
    labels: Dict[str, torch.Tensor],
    sample_i: int,
    diseases: Sequence[str],
) -> Tuple[Optional[str], int]:
    for d in diseases:
        y = labels[d][sample_i].item()
        if y >= 0:
            return d, y
    return None, -1


def head_abnormal_probability(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    t = max(float(temperature), 1e-6)
    probs = F.softmax(logits / t, dim=-1)
    return (1.0 - probs[0]).item()


def _device_bucket(device_name: str) -> str:
    if device_name == SMARTPHONE:
        return SMARTPHONE
    if device_name in MEDICAL_DEVICES:
        return "medical"
    return "other"


def optimize_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
) -> float:
    """단일 헤드 NLL 최소화 Temperature (LBFGS on log T)."""
    if logits.size(0) < 2:
        return 1.0

    log_t = torch.zeros(1, dtype=logits.dtype, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        loss = criterion(logits / t, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        return float(torch.exp(log_t).clamp(min=1e-3, max=100.0).item())
    except Exception:
        return 1.0


@torch.no_grad()
def collect_sample_records(
    model: nn.Module,
    loader: DataLoader,
    dataset: RandomSplitEyeDataset,
    diseases: List[str],
    device: str,
) -> List[SampleRecord]:
    model.eval()
    records: List[SampleRecord] = []
    local_idx = 0

    for images, labels in tqdm(loader, desc="Collect logits"):
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(dataset):
                break
            gt_disease, gt_label = _active_disease_and_label(labels, i, diseases)
            records.append(
                {
                    "device": dataset.get_device(local_idx),
                    "gt_disease": gt_disease,
                    "gt_label": gt_label,
                    "logits": {d: outputs[d][i].detach().cpu().clone() for d in diseases},
                    "labels": {d: labels[d][i].item() for d in diseases},
                }
            )
            local_idx += 1

    return records


def fit_per_head_temperatures(
    cal_records: List[SampleRecord],
    diseases: List[str],
) -> Dict[str, float]:
    """헤드별 독립 Temperature Scaling (활성 라벨 샘플만)."""
    temperatures: Dict[str, float] = {}
    print(f"\n{'=' * 64}")
    print("🌡️  Per-head Temperature Scaling (Calibration set)")
    print(f"{'=' * 64}")

    for d in diseases:
        logits_list: List[torch.Tensor] = []
        label_list: List[int] = []
        for rec in cal_records:
            y = rec["labels"][d]  # type: ignore[index]
            if y >= 0:
                logits_list.append(rec["logits"][d])  # type: ignore[index]
                label_list.append(int(y))

        n = len(logits_list)
        if n < 2:
            temperatures[d] = 1.0
            print(f"  {d:16s} T=1.0000  (n={n}, skip)")
            continue

        logits = torch.stack(logits_list)
        labels = torch.tensor(label_list, dtype=torch.long)
        t = optimize_temperature(logits, labels)
        temperatures[d] = t
        print(f"  {d:16s} T={t:.4f}  (n={n:,})")

    return temperatures


def _rank_diseases(
    record: SampleRecord,
    diseases: Sequence[str],
    temperatures: Dict[str, float],
    disease_weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    weights = disease_weights or {}
    scored = []
    for d in diseases:
        logits = record["logits"][d]  # type: ignore[index]
        prob = head_abnormal_probability(logits, temperatures.get(d, 1.0))
        scored.append((d, prob * weights.get(d, 1.0)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in scored]


def evaluate_topk_records(
    records: List[SampleRecord],
    diseases: List[str],
    temperatures: Dict[str, float],
    disease_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    hits = {k: 0 for k in TOP_KS}
    total = 0
    skipped_normal = 0

    per_disease_top3: Dict[str, Dict[str, int]] = {
        d: {"hit": 0, "total": 0} for d in diseases
    }
    subset_hits: Dict[str, Dict[int, int]] = {
        "all": {k: 0 for k in TOP_KS},
        SMARTPHONE: {k: 0 for k in TOP_KS},
        "medical": {k: 0 for k in TOP_KS},
    }
    subset_totals: Dict[str, int] = {"all": 0, SMARTPHONE: 0, "medical": 0}

    for rec in records:
        gt_disease = rec["gt_disease"]
        gt_label = rec["gt_label"]
        if gt_disease is None or gt_label is None or int(gt_label) <= 0:
            skipped_normal += 1
            continue

        gt_disease = str(gt_disease)
        bucket = _device_bucket(str(rec["device"]))
        ranked = _rank_diseases(rec, diseases, temperatures, disease_weights)

        per_disease_top3[gt_disease]["total"] += 1
        if gt_disease in ranked[:3]:
            per_disease_top3[gt_disease]["hit"] += 1

        for k in TOP_KS:
            if gt_disease in ranked[:k]:
                hits[k] += 1
                subset_hits["all"][k] += 1
                if bucket in (SMARTPHONE, "medical"):
                    subset_hits[bucket][k] += 1

        total += 1
        subset_totals["all"] += 1
        if bucket in (SMARTPHONE, "medical"):
            subset_totals[bucket] += 1

    def _acc(h: int, n: int) -> float:
        return h / n if n else 0.0

    device_subset = {}
    for name in ("all", SMARTPHONE, "medical"):
        n = subset_totals[name]
        device_subset[name] = {
            "n": n,
            **{f"top_{k}_acc": _acc(subset_hits[name][k], n) for k in TOP_KS},
        }

    per_disease = {
        d: {
            "top_3_hit_rate": _acc(v["hit"], v["total"]),
            "top_3_hits": v["hit"],
            "n": v["total"],
        }
        for d, v in per_disease_top3.items()
        if v["total"] > 0
    }

    return {
        "n_abnormal": total,
        "n_skipped_normal": skipped_normal,
        "topk_accuracy": {f"top_{k}": _acc(hits[k], total) for k in TOP_KS},
        "topk_hits": {f"top_{k}": hits[k] for k in TOP_KS},
        "device_subset": device_subset,
        "per_disease_top3_hit_rate": per_disease,
    }


def _print_topk_block(title: str, report: Dict[str, object]) -> None:
    n = int(report["n_abnormal"])
    print(f"\n  [{title}] 비정상 n={n:,}")
    for k in TOP_KS:
        acc = report["topk_accuracy"][f"top_{k}"]
        hits = report["topk_hits"][f"top_{k}"]
        print(f"    Top-{k}: {acc * 100:.2f}%  ({hits:,}/{n:,})")


def print_comparison(
    before: Dict[str, object],
    after: Dict[str, object],
    temperatures: Dict[str, float],
    diseases: List[str],
    cal_meta: Dict[str, object],
) -> None:
    print(f"\n{'=' * 64}")
    print("📊 Temperature Scaling Top-K 비교 (Test set)")
    print(f"{'=' * 64}")
    print(
        f"  Cal/Test split: {cal_meta['cal_size']:,} / {cal_meta['test_size']:,}  "
        f"(seed={cal_meta['cal_split_seed']}, group overlap={cal_meta['group_overlap']})"
    )

    print(f"\n{'=' * 64}")
    print("🌡️  학습된 Temperature (T=1.0 → 무보정)")
    print(f"{'=' * 64}")
    for d in diseases:
        print(f"  {d:16s} T={temperatures.get(d, 1.0):.4f}")

    _print_topk_block("Calibration 전 (T=1.0)", before)
    _print_topk_block("Calibration 후 (scaled)", after)

    print(f"\n  Δ Top-K (after − before):")
    for k in TOP_KS:
        b = before["topk_accuracy"][f"top_{k}"]
        a = after["topk_accuracy"][f"top_{k}"]
        print(f"    Top-{k}: {(a - b) * 100:+.2f}%p")

    print(f"\n{'=' * 64}")
    print("📱 Device subset (Test)")
    print(f"{'=' * 64}")
    for name, label in [("all", "전체"), (SMARTPHONE, "스마트폰"), ("medical", "의료장비")]:
        sub_b = before["device_subset"][name]
        sub_a = after["device_subset"][name]
        print(f"\n  [{label}] n={sub_b['n']:,}")
        for k in TOP_KS:
            vb = sub_b[f"top_{k}_acc"] * 100
            va = sub_a[f"top_{k}_acc"] * 100
            print(f"    Top-{k}: {vb:.2f}% → {va:.2f}%  ({va - vb:+.2f}%p)")

    print(f"\n{'=' * 64}")
    print("질환별 Top-3 hit rate (Test)")
    print(f"{'=' * 64}")
    print(f"  {'질환':16s} {'Before':>8s} {'After':>8s} {'Δ':>8s}")
    print("  " + "-" * 44)
    for d in diseases:
        if d not in before.get("per_disease_top3_hit_rate", {}):
            continue
        rb = before["per_disease_top3_hit_rate"][d]["top_3_hit_rate"] * 100
        ra = after["per_disease_top3_hit_rate"][d]["top_3_hit_rate"] * 100
        print(f"  {d:16s} {rb:7.2f}% {ra:7.2f}% {ra - rb:+7.2f}%p")


def save_results(
    animal_type: str,
    checkpoint: Path,
    split_meta: Dict[str, object],
    cal_meta: Dict[str, object],
    temperatures: Dict[str, float],
    before: Dict[str, object],
    after: Dict[str, object],
    disease_weights: Dict[str, float],
) -> Path:
    out_dir = Path(RandomSplitConfig.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"topk_calibrated_{animal_type}_{ts}.json"

    payload = {
        "animal_type": animal_type,
        "checkpoint": str(checkpoint),
        "timestamp": ts,
        "split_meta": split_meta,
        "cal_test_split": cal_meta,
        "temperatures": temperatures,
        "disease_weights": disease_weights,
        "before_calibration": before,
        "after_calibration": after,
        "delta_topk": {
            f"top_{k}": (
                after["topk_accuracy"][f"top_{k}"] - before["topk_accuracy"][f"top_{k}"]
            )
            for k in TOP_KS
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    animal_type = os.environ.get("ANIMAL_TYPE", "dog").strip().lower()
    if animal_type not in ("dog", "cat"):
        raise ValueError("ANIMAL_TYPE=dog 또는 cat")

    img_size = int(os.environ.get("IMG_SIZE", str(RandomSplitConfig.IMG_SIZE)))
    batch_size = resolve_batch_size(int(os.environ.get("BATCH_SIZE", "32")))
    num_workers = resolve_num_workers(int(os.environ.get("NUM_WORKERS", "4")))
    cal_ratio = float(os.environ.get("CAL_RATIO", "0.5"))
    cal_seed = int(os.environ.get("CAL_SPLIT_SEED", "43"))
    device = get_device()
    disease_weights = resolve_disease_weights()

    print("=" * 64)
    print(f"🔬 Top-K + Temperature Scaling ({animal_type.upper()})")
    print("=" * 64)
    print(f"  device: {device}")
    print(f"  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    print(f"  CAL_RATIO={cal_ratio}  CAL_SPLIT_SEED={cal_seed}")
    if disease_weights:
        print(f"  DISEASE_WEIGHTS={disease_weights}")

    checkpoint = resolve_checkpoint(animal_type)
    print(f"  checkpoint: {checkpoint}")

    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    diseases = model.get_disease_names()

    _, _, _, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal_type,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_sampler=False,
        mode="multitask",
    )

    cal_idx, test_idx, cal_meta = split_val_cal_test_indices(
        val_ds, cal_ratio=cal_ratio, seed=cal_seed,
    )
    print(
        f"\n  Val → Cal {len(cal_idx):,} / Test {len(test_idx):,}  "
        f"(crop_D* overlap={cal_meta['group_overlap']})"
    )

    cal_loader, cal_ds = make_subset_loader(val_ds, cal_idx, batch_size, num_workers)
    test_loader, test_ds = make_subset_loader(val_ds, test_idx, batch_size, num_workers)

    cal_records = collect_sample_records(model, cal_loader, cal_ds, diseases, device)
    temperatures = fit_per_head_temperatures(cal_records, diseases)

    test_records = collect_sample_records(model, test_loader, test_ds, diseases, device)
    unit_t = {d: 1.0 for d in diseases}

    before = evaluate_topk_records(test_records, diseases, unit_t, disease_weights)
    after = evaluate_topk_records(test_records, diseases, temperatures, disease_weights)

    print_comparison(before, after, temperatures, diseases, cal_meta)

    out_path = save_results(
        animal_type,
        checkpoint,
        split_meta,
        cal_meta,
        temperatures,
        before,
        after,
        disease_weights,
    )
    print(f"\n✓ 결과 저장: {out_path}")


if __name__ == "__main__":
    main()

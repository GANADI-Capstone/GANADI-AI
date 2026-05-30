"""
멀티태스크 random split 모델 Top-K 질환 분류 accuracy 평가.

TL Random Val (SPLIT_SEED=42, VAL_RATIO=0.2) 비정상 샘플만 사용.
각 질환 헤드의 P(비정상)=1-P(무)로 질환 순위를 매기고 Top-K hit rate 측정.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog python models/classifier/eval_multitask_topk.py
  ANIMAL_TYPE=cat python models/classifier/eval_multitask_topk.py

환경변수:
  ANIMAL_TYPE=dog|cat
  SPLIT_SEED=42
  VAL_RATIO=0.2
  CHECKPOINT=  — 기본 models/classifier/checkpoints/{animal}_best_random_split.pth
  BATCH_SIZE=32
  IMG_SIZE=300
"""

from __future__ import annotations

import json
import os
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
)
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers
from models.classifier.train_random_split import RandomSplitConfig

TOP_KS = (1, 2, 3, 5)


def resolve_checkpoint(animal_type: str) -> Path:
    raw = os.environ.get("CHECKPOINT", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"CHECKPOINT 없음: {path}")
        return path
    path = Path(RandomSplitConfig.best_checkpoint(animal_type))
    if not path.is_file():
        raise FileNotFoundError(
            f"체크포인트 없음: {path}\n"
            "  train_random_split.py 학습 후 실행하거나 CHECKPOINT= 지정"
        )
    return path


def head_abnormal_probability(logits: torch.Tensor) -> float:
    """헤드별 P(비정상) = 1 - P(클래스 0=무)."""
    probs = F.softmax(logits, dim=-1)
    return (1.0 - probs[0]).item()


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


def _rank_diseases_by_abnormal_prob(
    outputs: Dict[str, torch.Tensor],
    sample_i: int,
    diseases: Sequence[str],
) -> List[Tuple[str, float]]:
    scored = [
        (d, head_abnormal_probability(outputs[d][sample_i]))
        for d in diseases
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def _device_bucket(device_name: str) -> str:
    if device_name == SMARTPHONE:
        return SMARTPHONE
    if device_name in MEDICAL_DEVICES:
        return "medical"
    return "other"


@torch.no_grad()
def evaluate_multitask_topk(
    model: nn.Module,
    val_dataset: RandomSplitEyeDataset,
    val_loader: DataLoader,
    diseases: List[str],
    device: str,
) -> Dict[str, object]:
    model.eval()

    hits = {k: 0 for k in TOP_KS}
    total = 0

    per_disease_top3: Dict[str, Dict[str, int]] = {
        d: {"hit": 0, "total": 0} for d in diseases
    }
    cm: Dict[str, Dict[str, int]] = {d: {p: 0 for p in diseases} for d in diseases}

    subset_hits: Dict[str, Dict[int, int]] = {
        "all": {k: 0 for k in TOP_KS},
        SMARTPHONE: {k: 0 for k in TOP_KS},
        "medical": {k: 0 for k in TOP_KS},
    }
    subset_totals: Dict[str, int] = {"all": 0, SMARTPHONE: 0, "medical": 0}

    skipped_normal = 0
    local_idx = 0

    for images, labels in tqdm(val_loader, desc="Top-K eval"):
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(val_dataset):
                break

            gt_disease, gt_label = _active_disease_and_label(labels, i, diseases)
            dev_name = val_dataset.get_device(local_idx)
            bucket = _device_bucket(dev_name)

            if gt_disease is None or gt_label <= 0:
                skipped_normal += 1
                local_idx += 1
                continue

            ranked = _rank_diseases_by_abnormal_prob(outputs, i, diseases)
            ranked_names = [d for d, _ in ranked]
            pred_top1 = ranked_names[0]

            cm[gt_disease][pred_top1] += 1
            per_disease_top3[gt_disease]["total"] += 1
            if gt_disease in ranked_names[:3]:
                per_disease_top3[gt_disease]["hit"] += 1

            for k in TOP_KS:
                if gt_disease in ranked_names[:k]:
                    hits[k] += 1
                    subset_hits["all"][k] += 1
                    if bucket in (SMARTPHONE, "medical"):
                        subset_hits[bucket][k] += 1

            total += 1
            subset_totals["all"] += 1
            if bucket in (SMARTPHONE, "medical"):
                subset_totals[bucket] += 1

            local_idx += 1

    def _acc(h: int, n: int) -> float:
        return h / n if n else 0.0

    topk_acc = {f"top_{k}": _acc(hits[k], total) for k in TOP_KS}

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

    cm_matrix = [[cm[gt][pred] for pred in diseases] for gt in diseases]

    return {
        "n_abnormal": total,
        "n_skipped_normal": skipped_normal,
        "topk_accuracy": topk_acc,
        "topk_hits": {f"top_{k}": hits[k] for k in TOP_KS},
        "device_subset": device_subset,
        "per_disease_top3_hit_rate": per_disease,
        "confusion_matrix_top1": {
            "labels": diseases,
            "matrix": cm_matrix,
        },
    }


def print_results(report: Dict[str, object], diseases: List[str]) -> None:
    n = report["n_abnormal"]
    print(f"\n{'=' * 64}")
    print(f"📊 Top-K accuracy (비정상 n={n:,}, 정상 skip={report['n_skipped_normal']:,})")
    print(f"{'=' * 64}")
    for k in TOP_KS:
        acc = report["topk_accuracy"][f"top_{k}"]
        hits = report["topk_hits"][f"top_{k}"]
        print(f"  Top-{k}: {acc * 100:.2f}%  ({hits:,}/{n:,})")

    print(f"\n{'=' * 64}")
    print("📱 Device subset Top-K")
    print(f"{'=' * 64}")
    for name, label in [("all", "전체"), (SMARTPHONE, "스마트폰"), ("medical", "의료장비")]:
        sub = report["device_subset"][name]
        print(f"\n  [{label}] n={sub['n']:,}")
        for k in TOP_KS:
            print(f"    Top-{k}: {sub[f'top_{k}_acc'] * 100:.2f}%")

    print(f"\n{'=' * 64}")
    print("질환별 Top-3 hit rate")
    print(f"{'=' * 64}")
    for d in diseases:
        if d not in report["per_disease_top3_hit_rate"]:
            continue
        row = report["per_disease_top3_hit_rate"][d]
        print(
            f"  {d:16s} {row['top_3_hit_rate'] * 100:6.2f}%  "
            f"({row['top_3_hits']:,}/{row['n']:,})"
        )

    cm_info = report["confusion_matrix_top1"]
    labels: List[str] = cm_info["labels"]
    matrix: List[List[int]] = cm_info["matrix"]
    print(f"\n{'=' * 64}")
    print("Top-1 Confusion Matrix (행=실제, 열=예측)")
    print(f"{'=' * 64}")
    short = [n[:6] for n in labels]
    print("          " + " ".join(f"{s:>7s}" for s in short))
    for i, row in enumerate(matrix):
        print(f"{labels[i][:8]:>8s}  " + " ".join(f"{v:7d}" for v in row))


def save_results(
    animal_type: str,
    checkpoint: Path,
    split_meta: Dict[str, object],
    report: Dict[str, object],
) -> Path:
    out_dir = Path(RandomSplitConfig.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"topk_eval_{animal_type}_{ts}.json"

    payload = {
        "animal_type": animal_type,
        "checkpoint": str(checkpoint),
        "timestamp": ts,
        "split_meta": split_meta,
        **report,
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
    device = get_device()

    print("=" * 64)
    print(f"🔬 멀티태스크 Top-K 평가 ({animal_type.upper()})")
    print("=" * 64)
    print(f"  device: {device}")
    print(f"  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    print(f"  VAL_RATIO={os.environ.get('VAL_RATIO', '0.2')}")
    print("  평가 대상: TL Random Val · 비정상 샘플만")

    checkpoint = resolve_checkpoint(animal_type)
    print(f"  checkpoint: {checkpoint}")

    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    diseases = model.get_disease_names()
    print(f"  질환 헤드: {len(diseases)}개")

    _, val_loader, _, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal_type,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_sampler=False,
        mode="multitask",
    )

    report = evaluate_multitask_topk(model, val_ds, val_loader, diseases, device)
    print_results(report, diseases)

    out_path = save_results(animal_type, checkpoint, split_meta, report)
    print(f"\n✓ 결과 저장: {out_path}")


if __name__ == "__main__":
    main()

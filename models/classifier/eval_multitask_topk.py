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
  DISEASE_WEIGHTS='{"백내장":0.8}'  — Top-K 재정렬용 질환별 가중 (미명시=1.0)
  EXCLUDE_HEADS=백내장  — Top-K 경쟁에서 제외할 헤드 (쉼표 구분, 재학습 없이 시뮬레이션)

예시 (백내장 제외 시뮬레이션):
  CHECKPOINT=models/classifier/checkpoints/dog_best_random_split.pth \\
  EXCLUDE_HEADS=백내장 python models/classifier/eval_multitask_topk.py \\
    --animal dog --device cuda --batch-size 64
"""

from __future__ import annotations

import argparse
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
from models.classifier.inference_multitask import extract_state_dict
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers
from models.classifier.train_random_split import RandomSplitConfig

TOP_KS = (1, 2, 3, 5)


def resolve_exclude_heads(
    spec: Optional[str],
    diseases: Sequence[str],
) -> frozenset[str]:
    """EXCLUDE_HEADS — Top-K 경쟁에서 제외할 질환 헤드 (쉼표 구분)."""
    if not spec or not str(spec).strip():
        return frozenset()
    names = [x.strip() for x in str(spec).split(",") if x.strip()]
    unknown = [n for n in names if n not in diseases]
    if unknown:
        raise ValueError(f"알 수 없는 EXCLUDE_HEADS 질환: {unknown}")
    return frozenset(names)


def resolve_disease_weights(
    disease_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """DISEASE_WEIGHTS JSON — 질환별 후처리 가중 (기본 1.0)."""
    if disease_weights is not None:
        return {k: float(v) for k, v in disease_weights.items()}
    raw = os.environ.get("DISEASE_WEIGHTS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"DISEASE_WEIGHTS JSON 파싱 실패: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("DISEASE_WEIGHTS는 JSON 객체여야 합니다.")
    return {str(k): float(v) for k, v in data.items()}


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
    disease_weights: Optional[Dict[str, float]] = None,
    exclude_heads: Optional[frozenset[str]] = None,
) -> List[Tuple[str, float]]:
    weights = disease_weights or {}
    excluded = exclude_heads or frozenset()
    scored = []
    for d in diseases:
        if d in excluded:
            continue
        prob = head_abnormal_probability(outputs[d][sample_i])
        w = weights.get(d, 1.0)
        scored.append((d, prob * w))
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
    disease_weights: Optional[Dict[str, float]] = None,
    exclude_heads: Optional[frozenset[str]] = None,
) -> Dict[str, object]:
    model.eval()
    excluded = exclude_heads or frozenset()

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

            ranked = _rank_diseases_by_abnormal_prob(
                outputs, i, diseases, disease_weights, excluded,
            )
            if not ranked:
                local_idx += 1
                continue
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
        "disease_weights": disease_weights or {},
        "exclude_heads": sorted(excluded),
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
    weights = report.get("disease_weights") or {}
    excluded = report.get("exclude_heads") or []
    print(f"\n{'=' * 64}")
    print(f"📊 Top-K accuracy (비정상 n={n:,}, 정상 skip={report['n_skipped_normal']:,})")
    if excluded:
        print(f"  EXCLUDE_HEADS (Top-K 경쟁 제외): {excluded}")
    else:
        print("  EXCLUDE_HEADS: (미설정)")
    if weights:
        applied = {d: weights[d] for d in diseases if d in weights}
        print(f"  DISEASE_WEIGHTS 적용: {applied}")
    else:
        print("  DISEASE_WEIGHTS: (미적용, 전부 1.0)")
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="멀티태스크 Top-K 평가")
    p.add_argument("--animal", default=os.environ.get("ANIMAL_TYPE", "dog"), choices=["dog", "cat"])
    p.add_argument("--device", default=os.environ.get("DEVICE"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--exclude-heads", default=os.environ.get("EXCLUDE_HEADS", ""))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    animal_type = args.animal.strip().lower()

    img_size = int(os.environ.get("IMG_SIZE", str(RandomSplitConfig.IMG_SIZE)))
    env_bs = os.environ.get("BATCH_SIZE", "32")
    batch_size = resolve_batch_size(args.batch_size if args.batch_size is not None else int(env_bs))
    num_workers = resolve_num_workers(int(os.environ.get("NUM_WORKERS", "4")))
    device = args.device or get_device()

    print("=" * 64)
    print(f"🔬 멀티태스크 Top-K 평가 ({animal_type.upper()})")
    print("=" * 64)
    print(f"  device: {device}")
    print(f"  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    print(f"  VAL_RATIO={os.environ.get('VAL_RATIO', '0.2')}")
    print("  평가 대상: TL Random Val · 비정상 샘플만")

    disease_weights = resolve_disease_weights()
    if disease_weights:
        print(f"  DISEASE_WEIGHTS={disease_weights}")
    else:
        print("  DISEASE_WEIGHTS: (미설정)")

    checkpoint = resolve_checkpoint(animal_type)
    print(f"  checkpoint: {checkpoint}")

    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(ckpt))
    model = model.to(device)
    model.eval()
    diseases = model.get_disease_names()
    exclude_heads = resolve_exclude_heads(args.exclude_heads, diseases)
    if exclude_heads:
        print(f"  EXCLUDE_HEADS={sorted(exclude_heads)}")
    print(f"  질환 헤드: {len(diseases)}개")

    _, val_loader, _, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal_type,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_sampler=False,
        mode="multitask",
    )

    report = evaluate_multitask_topk(
        model, val_ds, val_loader, diseases, device, disease_weights, exclude_heads,
    )
    print_results(report, diseases)

    out_path = save_results(animal_type, checkpoint, split_meta, report)
    print(f"\n✓ 결과 저장: {out_path}")


if __name__ == "__main__":
    main()

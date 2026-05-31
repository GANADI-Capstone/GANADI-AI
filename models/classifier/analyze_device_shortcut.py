"""
Device shortcut 재검증 — device별 질환 변별력 분석.

Val 비정상 샘플을 촬영장비(스마트폰/검안경/일반카메라)별로 분리해
Top-1/Top-3·confusion·백내장 쏠림을 비교합니다.
백본 feature → device 분류 실험으로 shortcut 증거를 정량화합니다.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog python models/classifier/analyze_device_shortcut.py
  CHECKPOINT=models/classifier/checkpoints/dog_best_random_split.pth \\
  ANIMAL_TYPE=dog python models/classifier/analyze_device_shortcut.py \\
    --animal dog --device cuda --batch-size 64

환경변수:
  ANIMAL_TYPE=dog|cat
  SPLIT_SEED=42
  VAL_RATIO=0.2
  CHECKPOINT=
  BATCH_SIZE=32
  FOCUS_DISEASE=백내장  — 오분류 집계 대상 질환
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.dataset_random_split import (
    SMARTPHONE,
    RandomSplitEyeDataset,
    create_random_split_dataloaders,
)
from models.classifier.eval_multitask_topk import (
    TOP_KS,
    _active_disease_and_label,
    _rank_diseases_by_abnormal_prob,
    head_abnormal_probability,
    resolve_checkpoint,
)
from models.classifier.inference_multitask import extract_state_dict
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers
from models.classifier.train_random_split import RandomSplitConfig

CANONICAL_DEVICES = ("스마트폰", "검안경", "일반카메라")
SampleRecord = Dict[str, object]


def normalize_device(name: str) -> str:
    name = (name or "").strip()
    if name in CANONICAL_DEVICES:
        return name
    return name or "(unknown)"


@torch.no_grad()
def collect_records(
    model: nn.Module,
    loader: DataLoader,
    dataset: RandomSplitEyeDataset,
    diseases: List[str],
    device: str,
    *,
    extract_features: bool = False,
) -> List[SampleRecord]:
    model.eval()
    records: List[SampleRecord] = []
    local_idx = 0

    for images, labels in tqdm(loader, desc="Collect"):
        images = images.to(device)
        features = model.backbone(images) if extract_features else None
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(dataset):
                break
            gt_disease, gt_label = _active_disease_and_label(labels, i, diseases)
            dev = normalize_device(dataset.get_device(local_idx))
            probs = {d: head_abnormal_probability(outputs[d][i]) for d in diseases}
            ranked = _rank_diseases_by_abnormal_prob(outputs, i, diseases)
            ranked_names = [d for d, _ in ranked]

            rec: SampleRecord = {
                "device": dev,
                "gt_disease": gt_disease,
                "gt_label": gt_label,
                "probs": probs,
                "ranked": ranked_names,
                "pred_top1": ranked_names[0] if ranked_names else None,
            }
            if extract_features and features is not None:
                rec["feature"] = features[i].detach().cpu().numpy()
            records.append(rec)
            local_idx += 1

    return records


def abnormal_records(records: Sequence[SampleRecord]) -> List[SampleRecord]:
    out = []
    for rec in records:
        gt = rec["gt_disease"]
        y = rec["gt_label"]
        if gt is None or y is None or int(y) <= 0:
            continue
        out.append(rec)
    return out


def filter_by_device(
    records: Sequence[SampleRecord],
    device_name: str,
) -> List[SampleRecord]:
    return [r for r in records if r["device"] == device_name]


def evaluate_disease_metrics(
    records: Sequence[SampleRecord],
    diseases: List[str],
    focus_disease: str = "백내장",
) -> Dict[str, object]:
    """Top-K, confusion, focus_disease 오분류율."""
    cm = {d: {p: 0 for p in diseases} for d in diseases}
    totals = {d: 0 for d in diseases}
    top_hits = {k: 0 for k in TOP_KS}
    top3_hit_by_disease: Dict[str, Dict[str, int]] = {
        d: {"hit": 0, "total": 0} for d in diseases
    }
    focus_misclass = {d: {"mis": 0, "total": 0} for d in diseases if d != focus_disease}

    n = 0
    for rec in records:
        gt = str(rec["gt_disease"])
        ranked = rec["ranked"]  # type: ignore[assignment]
        pred = rec["pred_top1"]
        if pred is None or gt not in cm:
            continue

        n += 1
        cm[gt][str(pred)] += 1
        totals[gt] += 1

        for k in TOP_KS:
            if gt in ranked[:k]:
                top_hits[k] += 1
        top3_hit_by_disease[gt]["total"] += 1
        if gt in ranked[:3]:
            top3_hit_by_disease[gt]["hit"] += 1

        if gt != focus_disease and focus_disease in diseases:
            focus_misclass[gt]["total"] += 1
            if str(pred) == focus_disease:
                focus_misclass[gt]["mis"] += 1

    def acc(k: int) -> float:
        return top_hits[k] / n if n else 0.0

    per_disease_focus = {}
    for d, st in focus_misclass.items():
        if st["total"] > 0:
            per_disease_focus[d] = {
                "rate_to_focus": st["mis"] / st["total"],
                "mis": st["mis"],
                "total": st["total"],
            }

    focus_overall_mis = sum(st["mis"] for st in focus_misclass.values())
    focus_eligible = sum(st["total"] for st in focus_misclass.values())

    return {
        "n": n,
        "topk_accuracy": {f"top_{k}": acc(k) for k in TOP_KS},
        "topk_hits": {f"top_{k}": top_hits[k] for k in TOP_KS},
        "confusion_matrix": {gt: dict(row) for gt, row in cm.items()},
        "disease_totals": totals,
        "per_disease_top3": {
            d: {
                "hit_rate": v["hit"] / v["total"] if v["total"] else 0.0,
                **v,
            }
            for d, v in top3_hit_by_disease.items()
            if v["total"] > 0
        },
        "focus_disease": focus_disease,
        "focus_misclass_rate_overall": (
            focus_overall_mis / focus_eligible if focus_eligible else 0.0
        ),
        "focus_misclass_overall": focus_overall_mis,
        "focus_eligible_n": focus_eligible,
        "per_disease_to_focus": per_disease_focus,
    }


def analyze_by_device(
    val_abnormal: List[SampleRecord],
    diseases: List[str],
    focus_disease: str,
) -> Dict[str, object]:
    device_counts = Counter(r["device"] for r in val_abnormal)
    by_device: Dict[str, object] = {}

    for dev in CANONICAL_DEVICES:
        subset = filter_by_device(val_abnormal, dev)
        by_device[dev] = {
            "n_abnormal": len(subset),
            "metrics": evaluate_disease_metrics(subset, diseases, focus_disease),
        }

    other = [r for r in val_abnormal if r["device"] not in CANONICAL_DEVICES]
    if other:
        by_device["(other)"] = {
            "n_abnormal": len(other),
            "metrics": evaluate_disease_metrics(other, diseases, focus_disease),
        }

    return {
        "device_counts": dict(device_counts),
        "by_device": by_device,
        "all_abnormal": evaluate_disease_metrics(val_abnormal, diseases, focus_disease),
    }


def print_device_comparison(
    report: Dict[str, object],
    diseases: List[str],
    focus_disease: str,
) -> None:
    print(f"\n{'=' * 78}")
    print("Part 1 — Device별 Val 비정상 분포")
    print(f"{'=' * 78}")
    for dev, cnt in sorted(report["device_counts"].items(), key=lambda x: -x[1]):  # type: ignore[arg-type]
        print(f"  {dev:12s}  n={cnt:,}")

    print(f"\n{'=' * 78}")
    print(f"Part 2 — Device별 질환 변별력 (focus 오분류 → {focus_disease})")
    print(f"{'=' * 78}")
    header = (
        f"  {'Device':12s}  {'n':>7s}  {'Top-1':>7s}  {'Top-3':>7s}  "
        f"{'→'+focus_disease[:4]:>8s}  {'eligible':>9s}"
    )
    print(header)
    print("  " + "-" * 74)

    rows = []
    for dev in list(CANONICAL_DEVICES) + ["(other)"]:
        block = report["by_device"].get(dev)  # type: ignore[union-attr]
        if not block:
            continue
        m = block["metrics"]
        rows.append((dev, block["n_abnormal"], m))
        print(
            f"  {dev:12s}  {block['n_abnormal']:7,d}  "
            f"{m['topk_accuracy']['top_1'] * 100:6.2f}%  "
            f"{m['topk_accuracy']['top_3'] * 100:6.2f}%  "
            f"{m['focus_misclass_rate_overall'] * 100:7.2f}%  "
            f"{m['focus_eligible_n']:9,d}"
        )

    all_m = report["all_abnormal"]  # type: ignore[index]
    print("  " + "-" * 74)
    print(
        f"  {'전체':12s}  {all_m['n']:7,d}  "
        f"{all_m['topk_accuracy']['top_1'] * 100:6.2f}%  "
        f"{all_m['topk_accuracy']['top_3'] * 100:6.2f}%  "
        f"{all_m['focus_misclass_rate_overall'] * 100:7.2f}%  "
        f"{all_m['focus_eligible_n']:9,d}"
    )

    sp = report["by_device"].get("스마트폰", {}).get("metrics")  # type: ignore[union-attr]
    oph = report["by_device"].get("검안경", {}).get("metrics")  # type: ignore[union-attr]
    if sp and oph:
        diff1 = sp["topk_accuracy"]["top_1"] - oph["topk_accuracy"]["top_1"]
        diff3 = sp["topk_accuracy"]["top_3"] - oph["topk_accuracy"]["top_3"]
        print(f"\n  ★ 스마트폰 vs 검안경 Top-1 차이: {diff1 * 100:+.2f}%p")
        print(f"  ★ 스마트폰 vs 검안경 Top-3 차이: {diff3 * 100:+.2f}%p")
        if diff1 > 0:
            print("    → 스마트폰 비정상에서 Top-1이 더 높음 (device shortcut 약화 가능)")
        elif diff1 < 0:
            print("    → 검안경 비정상에서 Top-1이 더 높음 (의료장비 편향 가능)")

    print(f"\n{'=' * 78}")
    print(f"Part 3 — 질환별 → {focus_disease} 오분류율 (device별)")
    print(f"{'=' * 78}")
    print("  " + f"{'질환':12s}  " + "  ".join(f"{dev[:4]:>6s}" for dev in CANONICAL_DEVICES))

    non_focus = [d for d in diseases if d != focus_disease]
    for d in non_focus:
        cells = []
        for dev in CANONICAL_DEVICES:
            block = report["by_device"].get(dev)  # type: ignore[union-attr]
            if not block:
                cells.append("   —  ")
                continue
            pf = block["metrics"]["per_disease_to_focus"].get(d)  # type: ignore[index]
            if pf:
                cells.append(f"{pf['rate_to_focus'] * 100:5.1f}%")
            else:
                cells.append("   —  ")
        print(f"  {d:12s}  " + "  ".join(f"{c:>6s}" for c in cells))

    print(f"\n{'=' * 78}")
    print("Part 4 — Device별 Top-1 Confusion Matrix (10×10)")
    print(f"{'=' * 78}")
    for dev in CANONICAL_DEVICES:
        block = report["by_device"].get(dev)  # type: ignore[union-attr]
        if not block or block["n_abnormal"] == 0:
            continue
        cm = block["metrics"]["confusion_matrix"]  # type: ignore[index]
        print(f"\n  [{dev}] n={block['n_abnormal']:,}")
        short_l = [n[:5] for n in diseases]
        print("           " + " ".join(f"{s:>6s}" for s in short_l))
        for gt in diseases:
            row = cm.get(gt, {})
            if not sum(row.values()):
                continue
            print(
                f"  {gt[:8]:>8s}  "
                + " ".join(f"{row.get(p, 0):6d}" for p in diseases)
            )


def _majority_baseline(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    c = Counter(labels)
    return c.most_common(1)[0][1] / len(labels)


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, object]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "sklearn 미설치 — pip install scikit-learn"}

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_va = scaler.transform(x_val)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(x_tr, y_train)
    pred = clf.predict(x_va)
    acc = float((pred == y_val).mean())
    baseline = _majority_baseline(y_val.tolist())

    return {
        "accuracy": acc,
        "majority_baseline": baseline,
        "lift_over_majority": acc - baseline,
        "confusion_matrix": confusion_matrix(y_val, pred, labels=clf.classes_).tolist(),
        "classes": clf.classes_.tolist(),
        "classification_report": classification_report(y_val, pred, zero_division=0),
    }


def device_prediction_experiment(
    train_records: List[SampleRecord],
    val_records: List[SampleRecord],
    diseases: List[str],
) -> Dict[str, object]:
    """백본 feature / P(비정상) 벡터로 device 3-class 예측."""
    def _prepare(records: Sequence[SampleRecord], need_feature: bool) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for rec in records:
            dev = rec["device"]
            if dev not in CANONICAL_DEVICES:
                continue
            if need_feature:
                feat = rec.get("feature")
                if feat is None:
                    continue
                xs.append(feat)
            else:
                probs = rec["probs"]  # type: ignore[assignment]
                xs.append([probs[d] for d in diseases])
            ys.append(dev)
        if not xs:
            return np.zeros((0, 1)), np.array([])
        return np.stack(xs), np.array(ys)

    train_all = [r for r in train_records if r["device"] in CANONICAL_DEVICES]
    val_all = [r for r in val_records if r["device"] in CANONICAL_DEVICES]

    x_tr_f, y_tr = _prepare(train_all, need_feature=True)
    x_va_f, y_va = _prepare(val_all, need_feature=True)
    x_tr_p, _ = _prepare(train_all, need_feature=False)
    x_va_p, _ = _prepare(val_all, need_feature=False)

    results: Dict[str, object] = {
        "n_train": len(y_tr),
        "n_val": len(y_va),
        "val_device_distribution": dict(Counter(y_va.tolist())),
    }

    if len(y_tr) > 0 and len(y_va) > 0:
        results["backbone_logistic"] = _fit_logistic(x_tr_f, y_tr, x_va_f, y_va)
        results["p_abnormal_logistic"] = _fit_logistic(x_tr_p, y_tr, x_va_p, y_va)
    else:
        results["error"] = "feature/device 샘플 부족"

    return results


def print_device_prediction(report: Dict[str, object]) -> None:
    print(f"\n{'=' * 78}")
    print("Part 5 — Device 예측 실험 (Train fit → Val eval, 3-class)")
    print(f"{'=' * 78}")
    print(f"  Train n={report.get('n_train', 0):,}  Val n={report.get('n_val', 0):,}")
    print(f"  Val device 분포: {report.get('val_device_distribution', {})}")

    if report.get("error"):
        print(f"  ⚠ {report['error']}")
        return

    for name, label in [
        ("backbone_logistic", "백본 feature (1536-d) → LogisticRegression"),
        ("p_abnormal_logistic", "P(비정상) 벡터 (10-d) → LogisticRegression"),
    ]:
        block = report.get(name, {})
        if not block or block.get("error"):
            print(f"\n  [{label}]  {block.get('error', 'N/A')}")
            continue
        print(f"\n  [{label}]")
        print(f"    Val accuracy       : {block['accuracy'] * 100:.2f}%")
        print(f"    Majority baseline  : {block['majority_baseline'] * 100:.2f}%")
        print(f"    Lift over majority : {block['lift_over_majority'] * 100:+.2f}%p")
        if block["accuracy"] > 0.5:
            print("    ⚠ device를 쉽게 맞힘 → 백본/헤드에 device 정보 강하게 인코딩 (shortcut 증거)")
        print(block["classification_report"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Device별 질환 변별력 / shortcut 분석")
    p.add_argument("--animal", default=os.environ.get("ANIMAL_TYPE", "dog"), choices=["dog", "cat"])
    p.add_argument("--device", default=os.environ.get("DEVICE"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--focus-disease", default=os.environ.get("FOCUS_DISEASE", "백내장"))
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    animal = args.animal.strip().lower()
    torch_device = args.device or get_device()
    img_size = int(os.environ.get("IMG_SIZE", str(RandomSplitConfig.IMG_SIZE)))
    env_bs = os.environ.get("BATCH_SIZE", "32")
    batch_size = resolve_batch_size(args.batch_size if args.batch_size is not None else int(env_bs))
    num_workers = resolve_num_workers(int(os.environ.get("NUM_WORKERS", "4")))
    focus = args.focus_disease.strip()

    out_dir = Path(args.output_dir or RandomSplitConfig.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Device Shortcut 분석 — {animal.upper()}")
    print("=" * 78)
    print(f"  torch_device={torch_device}  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    print(f"  VAL_RATIO={os.environ.get('VAL_RATIO', '0.2')}  FOCUS_DISEASE={focus}")

    checkpoint = resolve_checkpoint(animal)
    print(f"  checkpoint: {checkpoint}")

    model = create_model(animal, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    model.load_state_dict(extract_state_dict(ckpt))
    model.to(torch_device)
    model.eval()
    diseases = model.get_disease_names()

    train_loader, val_loader, train_ds, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_sampler=False,
        mode="multitask",
    )

    val_records = collect_records(
        model, val_loader, val_ds, diseases, torch_device, extract_features=True,
    )
    val_abnormal = abnormal_records(val_records)

    part1 = analyze_by_device(val_abnormal, diseases, focus)
    print_device_comparison(part1, diseases, focus)

    train_records = collect_records(
        model, train_loader, train_ds, diseases, torch_device, extract_features=True,
    )
    part2 = device_prediction_experiment(train_records, val_records, diseases)
    print_device_prediction(part2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "animal_type": animal,
        "checkpoint": str(checkpoint),
        "timestamp": ts,
        "focus_disease": focus,
        "split_meta": split_meta,
        "device_disease_analysis": part1,
        "device_prediction": part2,
    }
    json_path = out_dir / f"device_shortcut_{animal}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ JSON 저장: {json_path}")


if __name__ == "__main__":
    main()

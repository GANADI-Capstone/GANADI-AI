"""
혼동 방향성 분석 + Split Conformal Prediction (멀티태스크 독립 헤드).

재학습 없이 기존 checkpoint로 Val 비정상 샘플을 분석합니다.
Top-1 경쟁은 eval_multitask_topk.py와 동일한 P(비정상) 순위를 사용합니다.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog python models/classifier/analyze_confusion_and_cp.py
  ANIMAL_TYPE=dog python models/classifier/analyze_confusion_and_cp.py \\
    --device cuda --batch-size 64 --mutual-threshold 0.10

환경변수:
  ANIMAL_TYPE=dog|cat
  SPLIT_SEED=42
  VAL_RATIO=0.2
  CAL_SPLIT_SEED=43   — Val 내 cal/test 분할 (Part B)
  CAL_RATIO=0.5
  CHECKPOINT=
  BATCH_SIZE=32
  MUTUAL_THRESHOLD=0.10 — 상호 혼동 쌍 임계값
  CP_ALPHAS=0.10,0.05   — 목표 miscoverage (coverage 90%, 95%)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.dataset_random_split import (
    RandomSplitEyeDataset,
    create_random_split_dataloaders,
)
from models.classifier.eval_multitask_topk import (
    _active_disease_and_label,
    _rank_diseases_by_abnormal_prob,
    head_abnormal_probability,
    resolve_checkpoint,
)
from models.classifier.eval_multitask_topk_calibrated import (
    make_subset_loader,
    split_val_cal_test_indices,
)
from models.classifier.inference_multitask import extract_state_dict
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers
from models.classifier.train_random_split import RandomSplitConfig

SampleRecord = Dict[str, object]


def parse_alphas(spec: str) -> List[float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [float(p) for p in parts]


def split_conformal_p_threshold(scores: List[float], alpha: float) -> float:
    """nonconformity s=1-P(abnormal) → P(abnormal) 임계값 (1-q)."""
    n = len(scores)
    if n == 0:
        return 0.5
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(k, n))
    q = sorted(scores)[k - 1]
    return 1.0 - q


@torch.no_grad()
def collect_inference_records(
    model: nn.Module,
    loader: DataLoader,
    dataset: RandomSplitEyeDataset,
    diseases: List[str],
    device: str,
) -> List[SampleRecord]:
    model.eval()
    records: List[SampleRecord] = []
    local_idx = 0

    for images, labels in tqdm(loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(dataset):
                break
            gt_disease, gt_label = _active_disease_and_label(labels, i, diseases)
            probs = {d: head_abnormal_probability(outputs[d][i]) for d in diseases}
            ranked = _rank_diseases_by_abnormal_prob(outputs, i, diseases)
            ranked_names = [d for d, _ in ranked]
            records.append(
                {
                    "gt_disease": gt_disease,
                    "gt_label": gt_label,
                    "probs": probs,
                    "ranked": ranked_names,
                    "pred_top1": ranked_names[0] if ranked_names else None,
                }
            )
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


def build_confusion_matrix(
    records: Sequence[SampleRecord],
    diseases: List[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    cm = {d: {p: 0 for p in diseases} for d in diseases}
    totals = {d: 0 for d in diseases}
    for rec in records:
        gt = str(rec["gt_disease"])
        pred = rec["pred_top1"]
        if pred is None or gt not in cm:
            continue
        cm[gt][str(pred)] += 1
        totals[gt] += 1
    return cm, totals


def compute_directional_rates(
    cm: Dict[str, Dict[str, int]],
    totals: Dict[str, int],
    diseases: List[str],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """질환쌍 (i,j) 양방향 혼동률."""
    pairs: Dict[Tuple[str, str], Dict[str, float]] = {}
    for i in diseases:
        for j in diseases:
            if i == j:
                continue
            ni = totals.get(i, 0)
            nj = totals.get(j, 0)
            i_to_j = cm[i][j] / ni if ni else 0.0
            j_to_i = cm[j][i] / nj if nj else 0.0
            pairs[(i, j)] = {
                "i_to_j": i_to_j,
                "j_to_i": j_to_i,
                "i_to_j_count": cm[i][j],
                "j_to_i_count": cm[j][i],
                "n_i": ni,
                "n_j": nj,
            }
    return pairs


def classify_confusion_pairs(
    cm: Dict[str, Dict[str, int]],
    totals: Dict[str, int],
    diseases: List[str],
    threshold: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """상호 혼동 vs 한방향 편향 쌍 분류."""
    mutual: List[Dict] = []
    asymmetric: List[Dict] = []
    unidirectional: List[Dict] = []

    for ii, i in enumerate(diseases):
        for j in diseases[ii + 1:]:
            ni, nj = totals.get(i, 0), totals.get(j, 0)
            rate_ij = cm[i][j] / ni if ni else 0.0
            rate_ji = cm[j][i] / nj if nj else 0.0
            entry = {
                "disease_a": i,
                "disease_b": j,
                "a_to_b": rate_ij,
                "b_to_a": rate_ji,
                "a_to_b_count": cm[i][j],
                "b_to_a_count": cm[j][i],
                "n_a": ni,
                "n_b": nj,
            }
            if rate_ij >= threshold and rate_ji >= threshold:
                mutual.append(entry)
            elif (rate_ij >= threshold) ^ (rate_ji >= threshold):
                entry["dominant"] = j if rate_ij >= rate_ji else i
                entry["absorbed"] = i if rate_ij >= rate_ji else j
                asymmetric.append(entry)

            if rate_ij > 0:
                unidirectional.append({
                    "from": i, "to": j, "rate": rate_ij,
                    "count": cm[i][j], "n": ni,
                })
            if rate_ji > 0:
                unidirectional.append({
                    "from": j, "to": i, "rate": rate_ji,
                    "count": cm[j][i], "n": nj,
                })

    mutual.sort(key=lambda x: -(x["a_to_b"] + x["b_to_a"]))
    asymmetric.sort(key=lambda x: -max(x["a_to_b"], x["b_to_a"]))
    unidirectional.sort(key=lambda x: -x["rate"])
    return mutual, asymmetric, unidirectional


def union_find_clusters(edges: List[Tuple[str, str]], nodes: List[str]) -> List[List[str]]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    groups: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return [sorted(g) for g in groups.values()]


def analyze_part_a(
    records: List[SampleRecord],
    diseases: List[str],
    threshold: float,
) -> Dict[str, object]:
    cm, totals = build_confusion_matrix(records, diseases)
    pairs = compute_directional_rates(cm, totals, diseases)
    mutual, asymmetric, unidirectional = classify_confusion_pairs(
        cm, totals, diseases, threshold,
    )

    mutual_edges = [(m["disease_a"], m["disease_b"]) for m in mutual]
    clusters = union_find_clusters(mutual_edges, diseases)

    return {
        "n_abnormal": len(records),
        "mutual_threshold": threshold,
        "confusion_matrix": {gt: dict(row) for gt, row in cm.items()},
        "disease_totals": totals,
        "directional_rates": {
            f"{i}->{j}": v for (i, j), v in pairs.items()
        },
        "mutual_pairs": mutual,
        "asymmetric_pairs": asymmetric,
        "top_directional_misclass": unidirectional[:20],
        "clusters": clusters,
    }


def print_part_a(report: Dict[str, object], diseases: List[str]) -> None:
    threshold = report["mutual_threshold"]
    print(f"\n{'=' * 72}")
    print(f"Part A — 혼동 방향성 (비정상 n={report['n_abnormal']:,}, 상호혼동 임계={threshold:.0%})")
    print(f"{'=' * 72}")

    totals = report["disease_totals"]
    print("\n[Top-1 오분류 Top-10 (단방향)]")
    for row in report["top_directional_misclass"][:10]:
        print(
            f"  {row['from']:12s} → {row['to']:12s}  "
            f"{row['rate'] * 100:5.1f}%  ({row['count']}/{row['n']})"
        )

    print(f"\n[상호 혼동 쌍] (양방향 ≥ {threshold:.0%}) — {len(report['mutual_pairs'])}쌍")
    for m in report["mutual_pairs"]:
        print(
            f"  {m['disease_a']:10s} ↔ {m['disease_b']:10s}  "
            f"{m['a_to_b'] * 100:5.1f}% / {m['b_to_a'] * 100:5.1f}%"
        )

    print(f"\n[한방향 편향 쌍] (한쪽만 ≥ {threshold:.0%}) — {len(report['asymmetric_pairs'])}쌍")
    for a in report["asymmetric_pairs"][:15]:
        print(
            f"  {a['absorbed']:10s} → {a['dominant']:10s}  "
            f"{max(a['a_to_b'], a['b_to_a']) * 100:5.1f}%  "
            f"(역방향 {min(a['a_to_b'], a['b_to_a']) * 100:5.1f}%)"
        )

    print("\n[상호 혼동 클러스터]")
    for i, cluster in enumerate(report["clusters"], 1):
        if len(cluster) >= 2:
            print(f"  Cluster {i}: {cluster}")
    singletons = [c[0] for c in report["clusters"] if len(c) == 1]
    if singletons:
        print(f"  고립 질환: {singletons}")


def plot_confusion_graph(
    report: Dict[str, object],
    diseases: List[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  (matplotlib 없음 — 그래프 생략)")
        return

    mutual = report["mutual_pairs"]
    asymmetric = report["asymmetric_pairs"]
    clusters = report["clusters"]

    cluster_id = {}
    for ci, cluster in enumerate(clusters):
        for d in cluster:
            cluster_id[d] = ci

    n = len(diseases)
    angles = [2 * math.pi * i / n for i in range(n)]
    pos = {d: (math.cos(a), math.sin(a)) for d, a in zip(diseases, angles)}

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.tab10

    for a in asymmetric:
        x0, y0 = pos[a["absorbed"]]
        x1, y1 = pos[a["dominant"]]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.5),
        )

    for m in mutual:
        da, db = m["disease_a"], m["disease_b"]
        x0, y0 = pos[da]
        x1, y1 = pos[db]
        w = (m["a_to_b"] + m["b_to_a"]) / 2
        ax.plot([x0, x1], [y0, y1], color="crimson", linewidth=2 + 8 * w, alpha=0.7)

    for d in diseases:
        x, y = pos[d]
        cid = cluster_id.get(d, 0)
        ax.scatter(x, y, s=400, c=[cmap(cid % 10)], zorder=5)
        ax.text(x, y, d[:4], ha="center", va="center", fontsize=8, fontweight="bold")

    handles = [
        mpatches.Patch(color="crimson", label="상호 혼동 (양방향)"),
        mpatches.Patch(color="gray", label="한방향 편향"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title("Confusion Graph (Top-1)")
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ 혼동 그래프: {out_path}")


def calibrate_per_head_cp(
    cal_records: List[SampleRecord],
    diseases: List[str],
    alpha: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """헤드별 독립 split conformal — P(비정상) 임계값."""
    thresholds: Dict[str, float] = {}
    n_cal: Dict[str, int] = {}

    for d in diseases:
        scores = []
        for rec in cal_records:
            if str(rec["gt_disease"]) != d:
                continue
            p = rec["probs"][d]  # type: ignore[index]
            scores.append(1.0 - float(p))
        n_cal[d] = len(scores)
        thresholds[d] = split_conformal_p_threshold(scores, alpha)

    return thresholds, n_cal


def global_cp_threshold(
    cal_records: List[SampleRecord],
    alpha: float,
) -> float:
    """전역 단일 P(비정상) 임계값 (정답 헤드 nonconformity 기준)."""
    scores = []
    for rec in cal_records:
        gt = str(rec["gt_disease"])
        p = rec["probs"][gt]  # type: ignore[index]
        scores.append(1.0 - float(p))
    return split_conformal_p_threshold(scores, alpha)


def prediction_set_per_head(
    probs: Dict[str, float],
    thresholds: Dict[str, float],
) -> Set[str]:
    return {d for d, p in probs.items() if p >= thresholds.get(d, 0.5)}


def prediction_set_global(probs: Dict[str, float], p_thresh: float) -> Set[str]:
    return {d for d, p in probs.items() if p >= p_thresh}


def evaluate_cp_on_test(
    test_records: List[SampleRecord],
    diseases: List[str],
    thresholds: Dict[str, float],
    alpha: float,
    *,
    global_thresh: Optional[float] = None,
) -> Dict[str, object]:
    covered = 0
    sizes: List[int] = []
    singleton = 0
    per_disease_sizes: Dict[str, List[int]] = defaultdict(list)
    per_disease_covered: Dict[str, int] = defaultdict(int)
    per_disease_n: Dict[str, int] = defaultdict(int)

    top1_hits = 0
    top3_hits = 0

    for rec in test_records:
        gt = str(rec["gt_disease"])
        probs = rec["probs"]  # type: ignore[assignment]
        ranked = rec["ranked"]  # type: ignore[assignment]

        if global_thresh is not None:
            pred_set = prediction_set_global(probs, global_thresh)
        else:
            pred_set = prediction_set_per_head(probs, thresholds)

        size = len(pred_set)
        sizes.append(size)
        if size == 1:
            singleton += 1
        if gt in pred_set:
            covered += 1
            per_disease_covered[gt] += 1
        per_disease_sizes[gt].append(size)
        per_disease_n[gt] += 1

        if ranked and ranked[0] == gt:
            top1_hits += 1
        if gt in ranked[:3]:
            top3_hits += 1

    n = len(test_records)
    coverage = covered / n if n else 0.0

    per_disease = {}
    for d in diseases:
        dn = per_disease_n.get(d, 0)
        if dn == 0:
            continue
        dsizes = per_disease_sizes[d]
        per_disease[d] = {
            "n": dn,
            "coverage": per_disease_covered[d] / dn,
            "mean_set_size": sum(dsizes) / dn,
            "median_set_size": sorted(dsizes)[dn // 2],
        }

    return {
        "target_alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": coverage,
        "mean_set_size": sum(sizes) / n if n else 0.0,
        "median_set_size": sorted(sizes)[n // 2] if n else 0,
        "singleton_rate": singleton / n if n else 0.0,
        "set_size_histogram": dict(sorted(
            {s: sizes.count(s) for s in set(sizes)}.items()
        )),
        "top1_accuracy": top1_hits / n if n else 0.0,
        "top3_accuracy": top3_hits / n if n else 0.0,
        "per_disease": per_disease,
        "n_test": n,
    }


def analyze_part_b(
    cal_records: List[SampleRecord],
    test_records: List[SampleRecord],
    diseases: List[str],
    alphas: List[float],
) -> Dict[str, object]:
    cal_abn = abnormal_records(cal_records)
    test_abn = abnormal_records(test_records)

    results: Dict[str, object] = {
        "n_cal_abnormal": len(cal_abn),
        "n_test_abnormal": len(test_abn),
        "per_head_cp": {},
        "global_cp": {},
    }

    for alpha in alphas:
        key = f"alpha_{alpha:.2f}"
        th_per_head, n_cal = calibrate_per_head_cp(cal_abn, diseases, alpha)
        eval_per = evaluate_cp_on_test(
            test_abn, diseases, th_per_head, alpha,
        )
        results["per_head_cp"][key] = {  # type: ignore[index]
            "thresholds": th_per_head,
            "n_cal_per_disease": n_cal,
            "test_metrics": eval_per,
        }

        g_thresh = global_cp_threshold(cal_abn, alpha)
        eval_global = evaluate_cp_on_test(
            test_abn, diseases, {}, alpha, global_thresh=g_thresh,
        )
        results["global_cp"][key] = {  # type: ignore[index]
            "p_threshold": g_thresh,
            "test_metrics": eval_global,
        }

    return results


def print_part_b(report: Dict[str, object], diseases: List[str]) -> None:
    print(f"\n{'=' * 72}")
    print(
        f"Part B — Conformal Prediction "
        f"(cal n={report['n_cal_abnormal']:,}, test n={report['n_test_abnormal']:,})"
    )
    print(f"{'=' * 72}")

    for mode, label in [("per_head_cp", "헤드별 독립 CP"), ("global_cp", "전역 단일 CP")]:
        print(f"\n--- {label} ---")
        block = report[mode]  # type: ignore[index]
        for key, data in block.items():
            alpha = float(key.replace("alpha_", ""))
            target_cov = 1.0 - alpha
            m = data["test_metrics"]
            print(f"\n  목표 coverage {target_cov:.0%} (α={alpha:.2f})")
            print(f"    Empirical coverage : {m['empirical_coverage'] * 100:.2f}%")
            print(f"    Mean set size      : {m['mean_set_size']:.2f}")
            print(f"    Median set size    : {m['median_set_size']}")
            print(f"    Singleton rate     : {m['singleton_rate'] * 100:.2f}%")
            print(f"    Top-1 accuracy     : {m['top1_accuracy'] * 100:.2f}%")
            print(f"    Top-3 accuracy     : {m['top3_accuracy'] * 100:.2f}%")
            print(f"    Set size histogram : {m['set_size_histogram']}")

    print(f"\n{'=' * 72}")
    print("Top-1 / Top-3 vs CP (헤드별 독립, test set)")
    print(f"{'=' * 72}")
    per_head = report["per_head_cp"]  # type: ignore[index]
    print(f"  {'목표Cov':>8s}  {'Top-1':>7s}  {'Top-3':>7s}  {'CP cov':>7s}  {'CP |S|':>7s}  {'|S|=1':>7s}")
    for key in sorted(per_head.keys()):
        alpha = float(key.replace("alpha_", ""))
        m = per_head[key]["test_metrics"]
        print(
            f"  {1 - alpha:7.0%}  {m['top1_accuracy'] * 100:6.2f}%  "
            f"{m['top3_accuracy'] * 100:6.2f}%  {m['empirical_coverage'] * 100:6.2f}%  "
            f"{m['mean_set_size']:7.2f}  {m['singleton_rate'] * 100:6.2f}%"
        )

    print("\n  질환별 CP mean set size (헤드별 α=0.10, coverage 90%)")
    key90 = "alpha_0.10"
    if key90 in per_head:
        pd = per_head[key90]["test_metrics"]["per_disease"]
        for d in diseases:
            if d not in pd:
                continue
            row = pd[d]
            print(
                f"    {d:16s}  |S|={row['mean_set_size']:.2f}  "
                f"cov={row['coverage'] * 100:.1f}%  (n={row['n']})"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="혼동 방향성 + Conformal Prediction 분석")
    p.add_argument("--animal", default=os.environ.get("ANIMAL_TYPE", "dog"), choices=["dog", "cat"])
    p.add_argument("--device", default=os.environ.get("DEVICE"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--mutual-threshold", type=float,
                   default=float(os.environ.get("MUTUAL_THRESHOLD", "0.10")))
    p.add_argument("--cal-split-seed", type=int,
                   default=int(os.environ.get("CAL_SPLIT_SEED", "43")))
    p.add_argument("--cal-ratio", type=float,
                   default=float(os.environ.get("CAL_RATIO", "0.5")))
    p.add_argument("--cp-alphas", default=os.environ.get("CP_ALPHAS", "0.10,0.05"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    animal = args.animal.strip().lower()
    device = args.device or get_device()
    img_size = int(os.environ.get("IMG_SIZE", str(RandomSplitConfig.IMG_SIZE)))
    env_bs = os.environ.get("BATCH_SIZE", "32")
    batch_size = resolve_batch_size(args.batch_size if args.batch_size is not None else int(env_bs))
    num_workers = resolve_num_workers(int(os.environ.get("NUM_WORKERS", "4")))
    alphas = parse_alphas(args.cp_alphas)

    out_dir = Path(args.output_dir or RandomSplitConfig.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"혼동 방향성 + Conformal Prediction — {animal.upper()}")
    print("=" * 72)
    print(f"  device={device}  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    print(f"  VAL_RATIO={os.environ.get('VAL_RATIO', '0.2')}")
    print(f"  CAL_SPLIT_SEED={args.cal_split_seed}  CAL_RATIO={args.cal_ratio}")
    print(f"  CP alphas={alphas}  mutual_threshold={args.mutual_threshold:.0%}")

    checkpoint = resolve_checkpoint(animal)
    print(f"  checkpoint: {checkpoint}")

    model = create_model(animal, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(ckpt))
    model.to(device)
    model.eval()
    diseases = model.get_disease_names()

    _, val_loader, _, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_sampler=False,
        mode="multitask",
    )

    all_records = collect_inference_records(model, val_loader, val_ds, diseases, device)
    val_abn = abnormal_records(all_records)

    part_a = analyze_part_a(val_abn, diseases, args.mutual_threshold)
    print_part_a(part_a, diseases)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_plot:
        plot_confusion_graph(
            part_a, diseases,
            out_dir / f"confusion_graph_{animal}_{ts}.png",
        )

    cal_idx, test_idx, cal_meta = split_val_cal_test_indices(
        val_ds, cal_ratio=args.cal_ratio, seed=args.cal_split_seed,
    )
    print(f"\n  Cal/Test split: cal={len(cal_idx):,} test={len(test_idx):,} "
          f"group_overlap={cal_meta['group_overlap']}")

    cal_loader, cal_ds = make_subset_loader(val_ds, cal_idx, batch_size, num_workers)
    test_loader, test_ds = make_subset_loader(val_ds, test_idx, batch_size, num_workers)
    cal_records = collect_inference_records(model, cal_loader, cal_ds, diseases, device)
    test_records = collect_inference_records(model, test_loader, test_ds, diseases, device)

    part_b = analyze_part_b(cal_records, test_records, diseases, alphas)
    print_part_b(part_b, diseases)

    payload = {
        "animal_type": animal,
        "checkpoint": str(checkpoint),
        "timestamp": ts,
        "split_meta": split_meta,
        "cal_split_meta": cal_meta,
        "part_a": part_a,
        "part_b": part_b,
    }
    json_path = out_dir / f"confusion_cp_{animal}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ JSON 저장: {json_path}")


if __name__ == "__main__":
    main()

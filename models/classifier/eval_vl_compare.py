"""
TL Random Split 모델을 VL(검증용 라벨링) 데이터로 추론하고
TL Random Val 결과와 비교합니다.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog python models/classifier/eval_vl_compare.py
  ANIMAL_TYPE=cat python models/classifier/eval_vl_compare.py

환경변수:
  ANIMAL_TYPE=dog|cat
  SPLIT_SEED=42          — TL random val split (train_random_split과 동일)
  VAL_RATIO=0.2
  CHECKPOINT=            — 기본: models/classifier/checkpoints/{animal}_best_random_split.pth
  BATCH_SIZE=32
  IMG_SIZE=300
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.dataset import EyeDiseaseDataset, get_transforms
from models.classifier.dataset_random_split import (
    MEDICAL_DEVICES,
    SMARTPHONE,
    RandomSplitEyeDataset,
    build_sample_metadata,
    create_random_split_dataloaders,
)
from models.classifier.model import create_model
from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers
from models.classifier.train_random_split import RandomSplitConfig, evaluate_device_subsets

DOG_VL_PATHS = ["eye_data/VL/개/안구/일반"]
CAT_VL_PATHS = ["eye_data/VL/고양이/안구/일반"]


class DeviceTaggedDataset(Protocol):
    def __len__(self) -> int: ...
    def get_device(self, i: int) -> str: ...


class VLEyeDataset(Dataset):
    """VL EyeDiseaseDataset + device 메타."""

    def __init__(self, base: EyeDiseaseDataset, device_by_index: Dict[int, str]):
        self.base = base
        self.device_by_index = device_by_index
        self.diseases = base.diseases

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        return self.base[i]

    def get_device(self, i: int) -> str:
        return self.device_by_index.get(i, "(unknown)")


def get_vl_paths(animal_type: str) -> List[str]:
    if animal_type.lower() == "dog":
        return list(DOG_VL_PATHS)
    if animal_type.lower() == "cat":
        return list(CAT_VL_PATHS)
    raise ValueError(f"animal_type은 'dog' 또는 'cat': {animal_type}")


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
            f"  train_random_split.py 학습 후 실행하거나 CHECKPOINT= 경로 지정"
        )
    return path


def load_history_device_report(animal_type: str) -> Optional[Dict[str, object]]:
    pattern = str(
        Path(RandomSplitConfig.OUTPUT_DIR)
        / f"training_history_{animal_type}_random_split_*.json"
    )
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    with open(files[0], encoding="utf-8") as f:
        data = json.load(f)
    return data.get("device_report")


@torch.no_grad()
def evaluate_per_disease(
    model: nn.Module,
    dataset: DeviceTaggedDataset,
    loader: DataLoader,
    diseases: Sequence[str],
    device: str,
    *,
    desc: str = "Per-disease eval",
) -> Dict[str, Dict[str, float]]:
    """질환별 accuracy (해당 질환 헤드가 활성인 샘플만)."""
    model.eval()
    correct: Dict[str, int] = {d: 0 for d in diseases}
    total: Dict[str, int] = {d: 0 for d in diseases}

    local_idx = 0
    for images, labels in tqdm(loader, desc=desc):
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(dataset):
                break
            for d in diseases:
                y = labels[d][i].item()
                if y < 0:
                    continue
                pred = torch.max(outputs[d][i], dim=0)[1].item()
                total[d] += 1
                correct[d] += int(pred == y)
            local_idx += 1

    return {
        d: {
            "accuracy": correct[d] / total[d] if total[d] else 0.0,
            "correct": correct[d],
            "total": total[d],
        }
        for d in diseases
    }


@dataclass
class EvalBundle:
    label: str
    device_report: Dict[str, object]
    per_disease: Dict[str, Dict[str, float]]


def create_vl_dataloader(
    animal_type: str,
    batch_size: int,
    img_size: int,
    num_workers: int,
) -> Tuple[DataLoader, VLEyeDataset]:
    vl_paths = get_vl_paths(animal_type)
    print(f"\n📁 VL 데이터: {vl_paths}")
    base = EyeDiseaseDataset(
        data_paths=vl_paths,
        animal_type=animal_type,
        transform=get_transforms(img_size, is_training=False),
        is_training=False,
    )
    metas = build_sample_metadata(base)
    device_map = {m.index: m.device for m in metas}
    ds = VLEyeDataset(base, device_map)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return loader, ds


def run_eval_bundle(
    model: nn.Module,
    dataset: DeviceTaggedDataset,
    loader: DataLoader,
    diseases: List[str],
    device: str,
    *,
    label: str,
) -> EvalBundle:
    device_report = evaluate_device_subsets(
        model, dataset, loader, diseases, device  # type: ignore[arg-type]
    )
    per_disease = evaluate_per_disease(
        model,
        dataset,
        loader,
        diseases,
        device,
        desc=f"{label} 질환별",
    )
    return EvalBundle(label=label, device_report=device_report, per_disease=per_disease)


def _pct(v: Optional[float], *, n: Optional[int] = None, show_n: bool = False) -> str:
    if v is None:
        return "—"
    text = f"{v * 100:.2f}%"
    if show_n and n is not None:
        text = f"{text} (n={n:,})"
    return text


def _get(report: Dict[str, object], key: str) -> Optional[float]:
    val = report.get(key)
    return float(val) if val is not None else None


def _get_count(report: Dict[str, object], key: str) -> int:
    counts = report.get("counts") or {}
    if isinstance(counts, dict):
        return int(counts.get(key, 0) or 0)
    return 0


def print_summary_table(tl: EvalBundle, vl: EvalBundle) -> None:
    rows = [
        ("전체 정확도", "val_mean_acc_all", "all", False),
        ("스마트폰 subset", "val_smartphone_acc", SMARTPHONE, False),
        ("스마트폰 비정상", "val_smartphone_abnormal_acc", "smartphone_abnormal", True),
        ("의료장비 subset", "val_medical_acc", "medical", False),
        ("Device 의존성", "device_dependency_score", None, False),
    ]

    w1, w2, w3 = 26, 15, 15
    top = f"╔{'═' * w1}╦{'═' * w2}╦{'═' * w3}╗"
    mid = f"╠{'═' * w1}╬{'═' * w2}╬{'═' * w3}╣"
    bot = f"╚{'═' * w1}╩{'═' * w2}╩{'═' * w3}╝"

    print(f"\n{top}")
    print(f"║ {'평가':<{w1 - 2}} ║ {'TL Random Val':^{w2 - 2}} ║ {'VL':^{w3 - 2}} ║")
    print(mid)

    for name, key, count_key, show_n in rows:
        tl_val = _get(tl.device_report, key)
        vl_val = _get(vl.device_report, key)
        tl_n = _get_count(tl.device_report, count_key) if count_key else None
        vl_n = _get_count(vl.device_report, count_key) if count_key else None
        tl_s = _pct(tl_val, n=tl_n, show_n=show_n and count_key == "smartphone_abnormal")
        vl_s = _pct(vl_val, n=vl_n, show_n=show_n and count_key == "smartphone_abnormal")
        print(f"║ {name:<{w1 - 2}} ║ {tl_s:>{w2 - 2}} ║ {vl_s:>{w3 - 2}} ║")

    print(bot)


def print_disease_table(
    tl: EvalBundle,
    vl: EvalBundle,
    diseases: Sequence[str],
) -> None:
    w1, w2, w3, w4 = 18, 12, 12, 10
    print(f"\n{'질환':<{w1}} {'TL Val Acc':>{w2}} {'VL Acc':>{w3}} {'Δ (VL-TL)':>{w4}}")
    print("-" * (w1 + w2 + w3 + w4 + 3))

    for d in diseases:
        tl_acc = tl.per_disease.get(d, {}).get("accuracy")
        vl_acc = vl.per_disease.get(d, {}).get("accuracy")
        tl_n = tl.per_disease.get(d, {}).get("total", 0)
        vl_n = vl.per_disease.get(d, {}).get("total", 0)

        if tl_acc is None and vl_acc is None:
            continue

        tl_s = f"{tl_acc * 100:.2f}%" if tl_acc is not None else "—"
        vl_s = f"{vl_acc * 100:.2f}%" if vl_acc is not None else "—"
        if tl_acc is not None and vl_acc is not None:
            delta = (vl_acc - tl_acc) * 100
            delta_s = f"{delta:+.2f}%"
        else:
            delta_s = "—"

        print(
            f"{d:<{w1}} {tl_s:>{w2}} {vl_s:>{w3}} {delta_s:>{w4}}  "
            f"(n={int(tl_n):,}/{int(vl_n):,})"
        )


def print_device_breakdown(report: Dict[str, object], title: str) -> None:
    accs = report.get("device_accuracies") or {}
    counts = report.get("counts") or {}
    if not isinstance(accs, dict):
        return
    print(f"\n  [{title}] device별 accuracy")
    for dev, acc in sorted(accs.items(), key=lambda x: (-x[1], x[0])):
        n = "?"
        if isinstance(counts, dict):
            # per_device counts not stored separately; skip exact n
            pass
        print(f"    {dev}: {acc * 100:.2f}%")


def save_results(
    animal_type: str,
    checkpoint: Path,
    tl: EvalBundle,
    vl: EvalBundle,
    history_report: Optional[Dict[str, object]],
    split_meta: Dict[str, object],
) -> Path:
    out_dir = Path("models/classifier/eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_vl_compare_{animal_type}_{ts}.json"

    payload = {
        "animal_type": animal_type,
        "checkpoint": str(checkpoint),
        "timestamp": ts,
        "split_meta": split_meta,
        "tl_random_val": {
            "device_report": tl.device_report,
            "per_disease": tl.per_disease,
        },
        "vl": {
            "paths": get_vl_paths(animal_type),
            "device_report": vl.device_report,
            "per_disease": vl.per_disease,
        },
        "training_history_device_report": history_report,
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
    print(f"🔬 TL Random Split → VL 비교 평가 ({animal_type.upper()})")
    print("=" * 64)
    print(f"  device: {device}")
    print(f"  SPLIT_SEED={os.environ.get('SPLIT_SEED', '42')}")
    print(f"  VAL_RATIO={os.environ.get('VAL_RATIO', '0.2')}")

    checkpoint = resolve_checkpoint(animal_type)
    print(f"  checkpoint: {checkpoint}")

    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    diseases = model.get_disease_names()
    print(f"  epoch={ckpt.get('epoch')}  val_acc_mean={ckpt.get('val_acc_mean')}")

    history_report = load_history_device_report(animal_type)
    if history_report:
        print("\n  (학습 history device_report 로드됨 — 아래 TL Val은 동일 split 재평가)")

    _, val_loader, _, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal_type,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_sampler=False,
    )

    print("\n" + "=" * 64)
    print("📊 TL Random Val 재평가")
    print("=" * 64)
    tl = run_eval_bundle(
        model, val_ds, val_loader, diseases, device, label="TL Random Val"
    )

    vl_loader, vl_ds = create_vl_dataloader(
        animal_type, batch_size, img_size, num_workers
    )

    print("\n" + "=" * 64)
    print("📊 VL 평가")
    print("=" * 64)
    vl = run_eval_bundle(model, vl_ds, vl_loader, diseases, device, label="VL")

    print("\n" + "=" * 64)
    print("📋 TL Random Val vs VL 비교")
    print("=" * 64)
    print_summary_table(tl, vl)
    print_disease_table(tl, vl, diseases)

    print_device_breakdown(tl.device_report, "TL Random Val")
    print_device_breakdown(vl.device_report, "VL")

    if history_report:
        hist_acc = history_report.get("val_mean_acc_all")
        if hist_acc is not None:
            print(
                f"\n  참고: 학습 종료 시 history val_mean_acc_all = {float(hist_acc) * 100:.2f}%"
            )

    out_path = save_results(
        animal_type, checkpoint, tl, vl, history_report, split_meta
    )
    print(f"\n✓ 결과 저장: {out_path}")


if __name__ == "__main__":
    main()

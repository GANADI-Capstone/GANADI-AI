"""
PyTorch vs ONNX FP32 vs ONNX INT8 벤치마크 (속도 + Top-K 정확도).

실행:
  python models/classifier/benchmark.py --animal dog
  python models/classifier/benchmark.py --animal cat --skip-accuracy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.inference_onnx import OnnxMultiTaskModel, load_onnx_model
from models.classifier.eval_multitask_topk import evaluate_multitask_topk
from models.classifier.export_onnx import onnx_path_for
from models.classifier.inference_multitask import (
    checkpoint_path_for,
    head_abnormal_probability,
    load_multitask_model,
)
from models.classifier.quantize_onnx import int8_path_for

from models.classifier.train import get_device, resolve_batch_size, resolve_num_workers

IMG_SIZE = 300


def file_size_mb(path: Path) -> float:
    if not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def measure_latency_ms(
    predict_fn: Callable[[torch.Tensor], object],
    device: str,
    n_images: int = 100,
    batch_size: int = 1,
) -> float:
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    if device not in ("cpu",) and not isinstance(predict_fn.__self__ if hasattr(predict_fn, "__self__") else None, OnnxMultiTaskModel):
        dummy = dummy.to(device)

    # warmup
    for _ in range(5):
        predict_fn(dummy)

    t0 = time.perf_counter()
    for _ in range(n_images):
        predict_fn(dummy)
    elapsed = time.perf_counter() - t0
    return elapsed / n_images * 1000


def load_pytorch(animal: str, device: str) -> nn.Module:
    model, _, _ = load_multitask_model(animal, device)
    return model


def binary_accuracy_on_val(model: nn.Module, animal: str, device: str) -> float:
    """비정상/정상 이진 판별 (any-head P(비정상) >= 0.5)."""
    from models.classifier.dataset_random_split import create_random_split_dataloaders

    _, val_loader, _, val_ds, _ = create_random_split_dataloaders(
        animal_type=animal,
        batch_size=resolve_batch_size(32),
        img_size=IMG_SIZE,
        num_workers=resolve_num_workers(2),
        use_sampler=False,
        mode="multitask",
    )
    diseases = model.get_disease_names()
    model.eval()
    correct = 0
    total = 0
    local_idx = 0
    is_onnx = isinstance(model, OnnxMultiTaskModel)

    with torch.no_grad():
        for images, labels in val_loader:
            if not is_onnx:
                images = images.to(device)
            outputs = model(images)
            bs = images.size(0)
            for i in range(bs):
                if local_idx >= len(val_ds):
                    break
                gt_abnormal = any(labels[d][i].item() > 0 for d in diseases)
                pred_abnormal = any(
                    head_abnormal_probability(outputs[d][i]) >= 0.5 for d in diseases
                )
                if pred_abnormal == gt_abnormal:
                    correct += 1
                total += 1
                local_idx += 1
    return correct / total if total else 0.0


def run_benchmark(animal: str, skip_accuracy: bool = False, n_latency: int = 100) -> None:
    animal = animal.lower()
    pt_device = get_device()
    cpu = "cpu"

    pth_path = checkpoint_path_for(animal)
    onnx_fp = onnx_path_for(animal)
    onnx_i8 = int8_path_for(animal)

    backends: List[Dict] = []

    # PyTorch
    if pth_path.is_file():
        pt_model = load_pytorch(animal, pt_device)
        pt_cpu_model = load_pytorch(animal, cpu) if pt_device != cpu else pt_model
        backends.append({
            "name": "PyTorch",
            "path": pth_path,
            "model": pt_model,
            "cpu_model": pt_cpu_model,
            "device": pt_device,
        })

    if onnx_fp.is_file():
        onnx_model, _ = load_onnx_model(animal, int8=False)
        backends.append({
            "name": "ONNX FP32",
            "path": onnx_fp,
            "model": onnx_model,
            "cpu_model": onnx_model,
            "device": "onnx-cpu",
        })

    if onnx_i8.is_file():
        onnx_i8_model, _ = load_onnx_model(animal, int8=True)
        backends.append({
            "name": "ONNX INT8",
            "path": onnx_i8,
            "model": onnx_i8_model,
            "cpu_model": onnx_i8_model,
            "device": "onnx-cpu",
        })

    if not backends:
        raise FileNotFoundError("벤치마크할 모델 없음. export_onnx.py / quantize_onnx.py 먼저 실행")

    print("=" * 72)
    print(f"Benchmark — {animal.upper()}")
    print("=" * 72)

    rows = []
    for b in backends:
        model = b["model"]
        cpu_model = b["cpu_model"]

        def _predict(m, x):
            m.eval()
            with torch.no_grad():
                return m(x)

        cpu_ms = measure_latency_ms(lambda x: _predict(cpu_model, x), cpu, n_images=n_latency)
        gpu_ms = None
        if b["device"] != "onnx-cpu" and b["device"] != cpu:
            gpu_ms = measure_latency_ms(lambda x: _predict(model, x.to(b["device"])), b["device"], n_images=n_latency)

        bin_acc = top3 = None
        if not skip_accuracy:
            from models.classifier.dataset_random_split import create_random_split_dataloaders

            _, val_loader, _, val_ds, _ = create_random_split_dataloaders(
                animal_type=animal,
                batch_size=resolve_batch_size(32),
                img_size=IMG_SIZE,
                num_workers=resolve_num_workers(2),
                use_sampler=False,
                mode="multitask",
            )
            diseases = model.get_disease_names()
            eval_device = cpu if isinstance(model, OnnxMultiTaskModel) else b["device"]
            report = evaluate_multitask_topk(
                model, val_ds, val_loader, diseases, eval_device,
            )
            top3 = report["topk_accuracy"]["top_3"]
            bin_acc = binary_accuracy_on_val(model, animal, eval_device)

        rows.append({
            "backend": b["name"],
            "size_mb": file_size_mb(b["path"]),
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "bin_acc": bin_acc,
            "top3": top3,
        })

    print(f"\n{'Backend':<14} {'Size':>8} {'CPU(ms)':>10} {'GPU(ms)':>10} {'Bin-Acc':>10} {'Top-3':>10}")
    print("-" * 72)
    baseline_top3 = rows[0].get("top3")
    for r in rows:
        gpu = f"{r['gpu_ms']:.0f}" if r["gpu_ms"] is not None else "N/A"
        bin_s = f"{r['bin_acc']*100:.2f}%" if r["bin_acc"] is not None else "-"
        top3_s = f"{r['top3']*100:.2f}%" if r["top3"] is not None else "-"
        if baseline_top3 and r["top3"] is not None and r["backend"] != "PyTorch":
            delta = (r["top3"] - baseline_top3) * 100
            top3_s += f" ({delta:+.2f}p)"
        print(
            f"{r['backend']:<14} {r['size_mb']:>7.0f}MB {r['cpu_ms']:>10.1f} {gpu:>10} "
            f"{bin_s:>10} {top3_s:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", default=os.environ.get("ANIMAL_TYPE", "dog"))
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--n-latency", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(args.animal, skip_accuracy=args.skip_accuracy, n_latency=args.n_latency)


if __name__ == "__main__":
    main()

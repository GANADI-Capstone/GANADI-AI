"""
ONNX FP32 → INT8 Quantization (Dynamic / Static).

실행:
  ANIMAL_TYPE=dog python models/classifier/quantize_onnx.py
  ANIMAL_TYPE=dog python models/classifier/quantize_onnx.py --static
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.export_onnx import IMG_SIZE, onnx_path_for
from models.classifier.inference_multitask import resolve_checkpoint_dir


def int8_path_for(animal_type: str) -> Path:
    return resolve_checkpoint_dir() / f"{animal_type}_best_random_split_int8.onnx"


def quantize_dynamic(animal_type: str) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    inp = onnx_path_for(animal_type)
    out = int8_path_for(animal_type)
    if not inp.is_file():
        raise FileNotFoundError(f"ONNX 없음: {inp}\n  먼저 export_onnx.py 실행")

    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        weight_type=QuantType.QUInt8,
    )
    _print_size(inp, out)
    return out


class ValCalibrationReader:
    """TL Random Val에서 calibration 샘플 수집."""

    def __init__(self, animal_type: str, max_samples: int = 200):
        import torch
        from torch.utils.data import DataLoader

        from models.classifier.dataset_random_split import create_random_split_dataloaders
        from models.classifier.train import resolve_num_workers

        _, val_loader, _, _, _ = create_random_split_dataloaders(
            animal_type=animal_type,
            batch_size=1,
            img_size=IMG_SIZE,
            num_workers=resolve_num_workers(2),
            use_sampler=False,
            mode="multitask",
        )
        self.input_name = "input"
        self._batches: List[np.ndarray] = []
        count = 0
        for images, _ in val_loader:
            self._batches.append(images.numpy())
            count += 1
            if count >= max_samples:
                break
        self._iter: Iterator[np.ndarray] | None = None

    def get_next(self) -> dict | None:
        if self._iter is None:
            self._iter = iter(self._batches)
        try:
            batch = next(self._iter)
        except StopIteration:
            return None
        return {self.input_name: batch}


def quantize_static(animal_type: str, calib_samples: int = 200) -> Path:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    inp = onnx_path_for(animal_type)
    out = int8_path_for(animal_type)
    if not inp.is_file():
        raise FileNotFoundError(f"ONNX 없음: {inp}")

    reader = ValCalibrationReader(animal_type, max_samples=calib_samples)

    class _Reader(CalibrationDataReader):
        def __init__(self, inner: ValCalibrationReader):
            self._inner = inner
            self._inner._iter = None

        def get_next(self):
            return self._inner.get_next()

    quantize_static(
        model_input=str(inp),
        model_output=str(out),
        calibration_data_reader=_Reader(reader),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    _print_size(inp, out)
    return out


def _print_size(inp: Path, out: Path) -> None:
    in_mb = inp.stat().st_size / (1024 * 1024)
    out_mb = out.stat().st_size / (1024 * 1024)
    ratio = (1 - out_mb / in_mb) * 100 if in_mb else 0
    print(f"✓ INT8 저장: {out}")
    print(f"  {in_mb:.1f} MB → {out_mb:.1f} MB ({ratio:.0f}% 감소)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX INT8 Quantization")
    parser.add_argument("--static", action="store_true", help="Static quantization (calibration)")
    parser.add_argument("--calib-samples", type=int, default=200)
    args = parser.parse_args()

    animal_type = os.environ.get("ANIMAL_TYPE", "dog").strip().lower()
    print("=" * 60)
    print(f"ONNX Quantize — {animal_type.upper()} ({'static' if args.static else 'dynamic'})")
    print("=" * 60)

    if args.static:
        quantize_static(animal_type, calib_samples=args.calib_samples)
    else:
        quantize_dynamic(animal_type)


if __name__ == "__main__":
    main()

"""
PyTorch 멀티태스크 체크포인트 → ONNX 변환.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog python models/classifier/export_onnx.py
  ANIMAL_TYPE=cat python models/classifier/export_onnx.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.classifier.inference_multitask import (
    checkpoint_path_for,
    extract_state_dict,
    resolve_checkpoint_dir,
)
from models.classifier.model import MultiTaskEyeDiseaseModel, create_model

OPSET_VERSION = 14
IMG_SIZE = 300


class MultiTaskOnnxWrapper(nn.Module):
    """Dict 출력 모델을 ONNX 호환 tuple 출력으로 래핑."""

    def __init__(self, model: MultiTaskEyeDiseaseModel):
        super().__init__()
        self.model = model
        self.disease_names: List[str] = model.get_disease_names()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(x)
        return tuple(outputs[d] for d in self.disease_names)


def output_names(disease_names: List[str]) -> List[str]:
    return [f"head_{d}" for d in disease_names]


def onnx_path_for(animal_type: str) -> Path:
    base = resolve_checkpoint_dir()
    return base / f"{animal_type}_best_random_split.onnx"


def load_model_for_export(animal_type: str, device: torch.device) -> MultiTaskEyeDiseaseModel:
    ckpt_path = checkpoint_path_for(animal_type)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    model = create_model(animal_type, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(ckpt))
    model.eval()
    return model


def export_onnx(
    animal_type: str,
    *,
    device: str = "cpu",
    verify: bool = True,
) -> Path:
    animal_type = animal_type.lower()
    dev = torch.device(device)
    model = load_model_for_export(animal_type, dev)
    wrapper = MultiTaskOnnxWrapper(model).to(dev).eval()

    diseases = wrapper.disease_names
    out_names = output_names(diseases)
    out_path = onnx_path_for(animal_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=out_names,
        dynamic_axes={"input": {0: "batch"}, **{n: {0: "batch"} for n in out_names}},
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✓ ONNX 저장: {out_path} ({size_mb:.1f} MB)")

    if verify:
        max_diff = verify_onnx(wrapper, out_path, dev, diseases, out_names)
        print(f"✓ PyTorch vs ONNX max diff: {max_diff:.2e}")
        if max_diff >= 1e-3:
            raise RuntimeError(f"ONNX 검증 실패: max diff {max_diff:.2e} >= 1e-3")

    _benchmark_export(dev, wrapper, out_path, diseases, out_names)
    return out_path


def verify_onnx(
    wrapper: MultiTaskOnnxWrapper,
    onnx_path: Path,
    device: torch.device,
    diseases: List[str],
    out_names: List[str],
) -> float:
    import onnxruntime as ort

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        pt_outs = wrapper(dummy)

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_outs = sess.run(out_names, {"input": dummy.cpu().numpy()})

    max_diff = 0.0
    for i, d in enumerate(diseases):
        diff = float(np.max(np.abs(pt_outs[i].cpu().numpy() - ort_outs[i])))
        max_diff = max(max_diff, diff)
        print(f"  {d:16s} max diff: {diff:.2e}")
    return max_diff


def _benchmark_export(
    device: torch.device,
    wrapper: MultiTaskOnnxWrapper,
    onnx_path: Path,
    diseases: List[str],
    out_names: List[str],
    n_warmup: int = 5,
    n_iter: int = 20,
) -> None:
    import onnxruntime as ort

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    np_in = dummy.cpu().numpy()

    with torch.no_grad():
        for _ in range(n_warmup):
            wrapper(dummy)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            wrapper(dummy)
        pt_ms = (time.perf_counter() - t0) / n_iter * 1000

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for _ in range(n_warmup):
        sess.run(out_names, {"input": np_in})
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sess.run(out_names, {"input": np_in})
    ort_ms = (time.perf_counter() - t0) / n_iter * 1000

    print(f"\n추론 속도 (batch=1, n={n_iter}):")
    print(f"  PyTorch ({device}): {pt_ms:.1f} ms")
    print(f"  ONNX CPU:           {ort_ms:.1f} ms")


def main() -> None:
    animal_type = os.environ.get("ANIMAL_TYPE", "dog").strip().lower()
    device = os.environ.get("EXPORT_DEVICE", "cpu")
    print("=" * 60)
    print(f"ONNX Export — {animal_type.upper()}")
    print("=" * 60)
    export_onnx(animal_type, device=device)


if __name__ == "__main__":
    main()

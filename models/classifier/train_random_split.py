"""
TL random split 멀티태스크 질환 분류 학습.

VL 대신 eye_data/개 + TL2 만 사용해 stratified random split (Val 20%).
기존 train.py / dataset.py 는 변경하지 않음.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog SPLIT_SEED=42 python models/classifier/train_random_split.py
  ANIMAL_TYPE=cat SPLIT_SEED=42 python models/classifier/train_random_split.py

환경변수:
  SPLIT_SEED=42
  VAL_RATIO=0.2
  USE_GROUP_SPLIT=1  — crop_D* 그룹 단위 분할
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.classifier.dataset_random_split import (
    MEDICAL_DEVICES,
    SMARTPHONE,
    RandomSplitEyeDataset,
    create_random_split_dataloaders,
)
from models.classifier.losses import build_per_disease_losses
from models.classifier.model import create_model, count_parameters
from models.classifier.train import (
    Config as BaseTrainConfig,
    _build_optimizer,
    _mean_disease_metric,
    _print_disease_metrics,
    _run_phase,
    get_device,
    resolve_batch_size,
    resolve_num_workers,
    train_epoch,
    validate_epoch,
)
from models.classifier.train_common import create_ema, ema_state_dict, eval_model


class RandomSplitConfig(BaseTrainConfig):
    SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
    VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
    USE_GROUP_SPLIT = os.environ.get("USE_GROUP_SPLIT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    @staticmethod
    def best_checkpoint(animal: str) -> str:
        return os.path.join(
            RandomSplitConfig.OUTPUT_DIR,
            f"{animal}_best_random_split.pth",
        )

    @staticmethod
    def final_checkpoint(animal: str) -> str:
        return os.path.join(
            RandomSplitConfig.OUTPUT_DIR,
            f"{animal}_final_random_split.pth",
        )


def _resolve_resume_path(config: RandomSplitConfig) -> str:
    raw = config.RESUME_CHECKPOINT
    if not raw or raw.lower() in ("0", "false", "no", "off"):
        return ""
    if raw.lower() == "auto":
        path = config.best_checkpoint(config.ANIMAL_TYPE)
        return path if os.path.isfile(path) else ""
    return raw if os.path.isfile(raw) else ""


@torch.no_grad()
def evaluate_device_subsets(
    model: nn.Module,
    val_dataset: RandomSplitEyeDataset,
    val_loader: DataLoader,
    diseases: List[str],
    device: str,
) -> Dict[str, object]:
    """Val device subset별 멀티태스크 정확도 (활성 질환 헤드 기준)."""
    model.eval()
    buckets = {
        "all": {"correct": 0, "total": 0},
        SMARTPHONE: {"correct": 0, "total": 0},
        "medical": {"correct": 0, "total": 0},
        "smartphone_abnormal": {"correct": 0, "total": 0},
    }
    per_device: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    local_idx = 0
    for images, labels in tqdm(val_loader, desc="Device subset eval"):
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)

        for i in range(bs):
            if local_idx >= len(val_dataset):
                break
            dev = val_dataset.get_device(local_idx)

            active_d = None
            active_y = -1
            for d in diseases:
                y = labels[d][i].item()
                if y >= 0:
                    active_d = d
                    active_y = y
                    break

            if active_d is not None:
                pred = torch.max(outputs[active_d][i], dim=0)[1].item()
                ok = pred == active_y

                buckets["all"]["total"] += 1
                buckets["all"]["correct"] += int(ok)
                per_device[dev]["total"] += 1
                per_device[dev]["correct"] += int(ok)

                if dev == SMARTPHONE:
                    buckets[SMARTPHONE]["total"] += 1
                    buckets[SMARTPHONE]["correct"] += int(ok)
                    if active_y > 0:
                        buckets["smartphone_abnormal"]["total"] += 1
                        buckets["smartphone_abnormal"]["correct"] += int(ok)
                elif dev in MEDICAL_DEVICES:
                    buckets["medical"]["total"] += 1
                    buckets["medical"]["correct"] += int(ok)

            local_idx += 1

    def _acc(cell: Dict[str, int]) -> float:
        return cell["correct"] / cell["total"] if cell["total"] else 0.0

    device_accs = {
        dev: _acc(v) for dev, v in sorted(per_device.items()) if v["total"] > 0
    }
    acc_values = list(device_accs.values())
    dependency = max(acc_values) - min(acc_values) if len(acc_values) >= 2 else 0.0

    report = {
        "val_mean_acc_all": _acc(buckets["all"]),
        "val_smartphone_acc": _acc(buckets[SMARTPHONE]),
        "val_medical_acc": _acc(buckets["medical"]),
        "val_smartphone_abnormal_acc": _acc(buckets["smartphone_abnormal"]),
        "device_accuracies": device_accs,
        "device_dependency_score": dependency,
        "counts": {
            "all": buckets["all"]["total"],
            SMARTPHONE: buckets[SMARTPHONE]["total"],
            "medical": buckets["medical"]["total"],
            "smartphone_abnormal": buckets["smartphone_abnormal"]["total"],
        },
    }

    print(f"\n{'=' * 64}")
    print("📱 Val device subset 평가 (활성 질환 헤드 accuracy)")
    print(f"{'=' * 64}")
    print(f"  전체:              acc={report['val_mean_acc_all']:.4f}  n={report['counts']['all']:,}")
    print(
        f"  스마트폰 (핵심):     acc={report['val_smartphone_acc']:.4f}  "
        f"n={report['counts'][SMARTPHONE]:,}"
    )
    print(
        f"  스마트폰+비정상:     acc={report['val_smartphone_abnormal_acc']:.4f}  "
        f"n={report['counts']['smartphone_abnormal']:,}"
    )
    print(
        f"  의료장비:            acc={report['val_medical_acc']:.4f}  "
        f"n={report['counts']['medical']:,}"
    )
    for dev, acc in device_accs.items():
        print(f"    [{dev}] acc={acc:.4f}  n={per_device[dev]['total']:,}")
    print(f"  Device 의존성 (max-min acc): {dependency:.4f}  ← 낮을수록 좋음")

    return report


def train():
    config = RandomSplitConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 64)
    print("멀티태스크 질환 분류 — TL Random Split 학습")
    print("=" * 64)
    print(f"  VL 미사용 | SPLIT_SEED={config.SPLIT_SEED} | VAL_RATIO={config.VAL_RATIO}")
    print(f"  USE_GROUP_SPLIT={'ON' if config.USE_GROUP_SPLIT else 'OFF'}")

    device = get_device()
    batch_size = resolve_batch_size(config.BATCH_SIZE)
    num_workers = resolve_num_workers(device, config.NUM_WORKERS)
    use_amp = device == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    use_sampler = config.ANIMAL_TYPE == "cat" and config.USE_SAMPLER_FOR_CAT
    boost_dis = config.SAMPLER_BOOST_DISEASE if use_sampler else None

    print(f"\n⚙️  {config.ANIMAL_TYPE.upper()} | Batch {batch_size} | AMP {'ON' if use_amp else 'OFF'}")
    print(f"  Phase1: {config.PHASE1_EPOCHS}ep  Phase2: {config.PHASE2_EPOCHS}ep")
    print(f"  Sampler: {'ON (고양이)' if use_sampler else 'OFF (강아지)'}")

    pin_mem = device == "cuda"
    train_loader, val_loader, train_ds, val_ds, split_meta = create_random_split_dataloaders(
        config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        num_workers=num_workers,
        use_sampler=use_sampler,
        sampler_boost_disease=boost_dis,
        sampler_boost_factor=config.SAMPLER_BOOST_FACTOR,
        pin_memory=pin_mem,
    )

    model = create_model(
        animal_type=config.ANIMAL_TYPE,
        pretrained=True,
        head_dropout=config.HEAD_DROPOUT,
    ).to(device)
    ema = create_ema(model, device)

    best_path = config.best_checkpoint(config.ANIMAL_TYPE)
    resume_path = _resolve_resume_path(config)
    skip_phase1 = False
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.module.load_state_dict(ckpt["model_state_dict"])
        skip_phase1 = True
        print(f"✓ 이어 학습: {resume_path}")

    print(f"학습 가능 파라미터: {count_parameters(model):,}")

    diseases = model.get_disease_names()
    criterion_dict = build_per_disease_losses(
        train_ds,
        diseases,
        config.LOSS_TYPE,
        device=device,
        use_class_weights=config.USE_CLASS_WEIGHTS,
        focal_gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING,
    ).to(device)

    history: dict = {
        "task": "multitask_random_split",
        "animal_type": config.ANIMAL_TYPE,
        "split_meta": split_meta,
        "train_loss": [],
        "val_loss": [],
        "train_acc_mean": [],
        "val_acc_mean": [],
        "acc_gap": [],
        "epochs": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0

    print("\n🚀 학습 시작...\n")

    if not skip_phase1 and config.PHASE1_EPOCHS > 0:
        optimizer = _build_optimizer(model, phase=1, config=config)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.PHASE1_EPOCHS)
        best_val_loss, patience_counter, best_path, last_epoch = _run_phase(
            phase=1,
            epochs=config.PHASE1_EPOCHS,
            global_epoch_start=0,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_dict=criterion_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            diseases=diseases,
            config=config,
            use_amp=use_amp,
            scaler=scaler,
            history=history,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_path=best_path,
            ema=ema,
        )
        if patience_counter < config.PATIENCE:
            patience_counter = 0

    if patience_counter < config.PATIENCE and config.PHASE2_EPOCHS > 0:
        optimizer = _build_optimizer(model, phase=2, config=config)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.PHASE2_EPOCHS)
        best_val_loss, patience_counter, best_path, last_epoch = _run_phase(
            phase=2,
            epochs=config.PHASE2_EPOCHS,
            global_epoch_start=last_epoch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_dict=criterion_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            diseases=diseases,
            config=config,
            use_amp=use_amp,
            scaler=scaler,
            history=history,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_path=best_path,
            ema=ema,
        )

    infer = eval_model(model, ema)
    device_report = evaluate_device_subsets(
        infer, val_ds, val_loader, diseases, device
    )
    history["device_report"] = device_report

    final_path = config.final_checkpoint(config.ANIMAL_TYPE)
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": ema_state_dict(ema, model),
            "best_val_loss": best_val_loss,
            "split_meta": split_meta,
            "task": "multitask_random_split",
            "animal_type": config.ANIMAL_TYPE,
            "ema_decay": 0.9998,
            "device_report": device_report,
        },
        final_path,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(
        config.OUTPUT_DIR,
        f"training_history_{config.ANIMAL_TYPE}_random_split_{ts}.json",
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 64)
    print("✅ TL Random Split 학습 완료")
    print("=" * 64)
    print(f"  Best:  {best_path}")
    print(f"  Final: {final_path}")
    print(f"  History: {history_path}")


if __name__ == "__main__":
    train()

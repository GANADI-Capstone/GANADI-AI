"""
정상 vs 비정상(통합 이진) 분류 모델 학습

실행 (프로젝트 루트):
  ANIMAL_TYPE=cat python models/classifier/train_binary.py
  ANIMAL_TYPE=dog python models/classifier/train_binary.py

Colab:
  !ANIMAL_TYPE=cat python models/classifier/train_binary.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models.classifier.dataset_binary import create_binary_dataloader
from models.classifier.model_binary import create_binary_model, count_parameters
from models.classifier.losses import FocalLoss


class Config:
    DOG_DATA_PATHS = [
        "eye_data/개/안구/일반",
        "eye_data/TL2/개/안구/일반",
    ]
    CAT_DATA_PATHS = [
        "eye_data/TL2/고양이/안구/일반",
    ]
    DOG_VAL_PATHS = ["eye_data/VL/개/안구/일반"]
    CAT_VAL_PATHS = ["eye_data/VL/고양이/안구/일반"]

    ANIMAL_TYPE = os.environ.get("ANIMAL_TYPE", "cat").strip().lower()

    IMG_SIZE = 300
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))

    PHASE1_EPOCHS = int(os.environ.get("PHASE1_EPOCHS", "4"))
    PHASE2_EPOCHS = int(os.environ.get("PHASE2_EPOCHS", "12"))
    HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
    FINETUNE_LR = float(os.environ.get("FINETUNE_LR", "1e-5"))

    WEIGHT_DECAY = 1e-4
    HEAD_DROPOUT = 0.4
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    USE_CLASS_WEIGHTS = True
    USE_SAMPLER = True

    PATIENCE = 5
    NUM_WORKERS = 0
    OUTPUT_DIR = "models/classifier/checkpoints"
    RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "auto").strip()


def get_device() -> str:
    if os.environ.get("FORCE_CPU_TRAINING", "").strip().lower() in ("1", "true", "yes"):
        print("⚠ CPU 학습 모드 (FORCE_CPU_TRAINING=1)")
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA"
        print(f"✓ CUDA ({name})")
        return "cuda"
    if torch.backends.mps.is_available():
        print("✓ MPS")
        return "mps"
    print("⚠ CPU")
    return "cpu"


def resolve_batch_size(configured: int) -> int:
    env_bs = os.environ.get("BATCH_SIZE", "").strip()
    if env_bs.isdigit() and int(env_bs) > 0:
        return int(env_bs)
    return configured if configured > 0 else 16


def resolve_num_workers(device: str, configured: int = 0) -> int:
    env_nw = os.environ.get("NUM_WORKERS", "").strip()
    if env_nw.isdigit():
        return int(env_nw)
    if configured and configured > 0:
        return configured
    return 2 if device == "cuda" else 0


def _checkpoint_name(animal: str, kind: str = "best") -> str:
    return os.path.join(Config.OUTPUT_DIR, f"{animal}_binary_{kind}.pth")


def _resolve_resume_path(config: Config) -> str:
    raw = config.RESUME_CHECKPOINT
    if not raw or raw.lower() in ("0", "false", "no", "off"):
        return ""
    if raw.lower() == "auto":
        path = _checkpoint_name(config.ANIMAL_TYPE, "best")
        return path if os.path.isfile(path) else ""
    return raw if os.path.isfile(raw) else ""


def _compute_binary_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """비정상(1)을 positive class로 Accuracy/Recall/Precision/F1."""
    preds = preds.view(-1)
    labels = labels.view(-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    total = tp + fp + fn + tn

    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    *,
    optimizer: Optional[optim.Optimizer] = None,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    grad_accum_steps: int = 1,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    desc = "Training" if is_train else "Validation"
    progress = tqdm(dataloader, desc=desc)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            with autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum_steps

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0 or step == len(dataloader):
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum_steps
        else:
            with torch.no_grad(), autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        progress.set_postfix({"loss": loss.item()})

    metrics = _compute_binary_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
    )
    avg_loss = total_loss / max(len(dataloader), 1)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    print(
        f"{prefix}  Loss: {metrics['loss']:.4f}  "
        f"Acc: {metrics['acc']:.4f}  "
        f"Recall: {metrics['recall']:.4f}  "
        f"Precision: {metrics['precision']:.4f}  "
        f"F1: {metrics['f1']:.4f}  "
        f"(positive=비정상)"
    )


def _build_optimizer(model: nn.Module, phase: int, config: Config) -> optim.Optimizer:
    if phase == 1:
        model.freeze_backbone()
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY)

    model.unfreeze_backbone()
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": config.FINETUNE_LR},
            {"params": head_params, "lr": config.FINETUNE_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def _run_phase(
    *,
    phase: int,
    epochs: int,
    global_epoch_start: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: str,
    config: Config,
    use_amp: bool,
    scaler: Optional[GradScaler],
    history: dict,
    best_val_loss: float,
    patience_counter: int,
    best_path: str,
) -> Tuple[float, int, str, int]:
    print(f"\n{'=' * 60}")
    print(f"Phase {phase}: {'헤드만 (freeze)' if phase == 1 else '전체 미세조정 (unfreeze)'}")
    print(f"{'=' * 60}")

    last_epoch = global_epoch_start

    for local_ep in range(1, epochs + 1):
        global_epoch = global_epoch_start + local_ep
        last_epoch = global_epoch
        print(f"\nEpoch {global_epoch} (Phase {phase} {local_ep}/{epochs})")
        print("-" * 60)

        _, train_m = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
        )
        _, val_m = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
        )
        scheduler.step()

        acc_gap = train_m["acc"] - val_m["acc"]
        _print_metrics("[Train]", train_m)
        _print_metrics("[Val]  ", val_m)
        print(f"[Gap]  Train-Val Acc: {acc_gap:+.4f}")

        record = {
            "epoch": global_epoch,
            "phase": phase,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_acc": train_m["acc"],
            "val_acc": val_m["acc"],
            "train_recall": train_m["recall"],
            "val_recall": val_m["recall"],
            "train_precision": train_m["precision"],
            "val_precision": val_m["precision"],
            "train_f1": train_m["f1"],
            "val_f1": val_m["f1"],
            "acc_gap": acc_gap,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history["epochs"].append(record)
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["acc"])
        history["val_acc"].append(val_m["acc"])
        history["acc_gap"].append(acc_gap)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_m["loss"],
                    "val_acc": val_m["acc"],
                    "val_f1": val_m["f1"],
                    "task": "binary_normal_abnormal",
                    "animal_type": config.ANIMAL_TYPE,
                },
                best_path,
            )
            print(f"✓ Best 저장: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n⚠ Early Stopping (patience={config.PATIENCE})")
                return best_val_loss, patience_counter, best_path, last_epoch

    return best_val_loss, patience_counter, best_path, last_epoch


def train():
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("정상 vs 비정상 이진 분류 학습")
    print("=" * 60)

    device = get_device()
    batch_size = resolve_batch_size(config.BATCH_SIZE)
    num_workers = resolve_num_workers(device, config.NUM_WORKERS)
    use_amp = device == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    if config.ANIMAL_TYPE == "dog":
        train_paths, val_paths = config.DOG_DATA_PATHS, config.DOG_VAL_PATHS
    else:
        train_paths, val_paths = config.CAT_DATA_PATHS, config.CAT_VAL_PATHS

    print(f"\n⚙️  {config.ANIMAL_TYPE.upper()} | Batch {batch_size} | AMP {'ON' if use_amp else 'OFF'}")
    print(f"  Phase1: {config.PHASE1_EPOCHS}ep  Phase2: {config.PHASE2_EPOCHS}ep")

    pin_mem = device == "cuda"
    train_loader = create_binary_dataloader(
        train_paths,
        config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        is_training=True,
        num_workers=num_workers,
        use_sampler=config.USE_SAMPLER,
        pin_memory=pin_mem,
    )
    val_loader = create_binary_dataloader(
        val_paths,
        config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        is_training=False,
        num_workers=num_workers,
        use_sampler=False,
        pin_memory=pin_mem,
    )

    model = create_binary_model(
        animal_type=config.ANIMAL_TYPE,
        head_dropout=config.HEAD_DROPOUT,
    ).to(device)

    resume_path = _resolve_resume_path(config)
    skip_phase1 = False
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        skip_phase1 = True
        print(f"✓ 이어 학습: {resume_path} (epoch {ckpt.get('epoch', '?')})")

    print(f"학습 가능 파라미터: {count_parameters(model):,}")

    alpha = None
    if config.USE_CLASS_WEIGHTS:
        cw = train_loader.dataset.get_class_weights().to(device)
        alpha = cw / cw.mean().clamp(min=1e-6)

    criterion = FocalLoss(
        gamma=config.FOCAL_GAMMA,
        alpha=alpha,
        label_smoothing=config.LABEL_SMOOTHING,
    )

    history = {
        "task": "binary_normal_abnormal",
        "animal_type": config.ANIMAL_TYPE,
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "acc_gap": [],
        "epochs": [],
    }

    best_path = _checkpoint_name(config.ANIMAL_TYPE, "best")
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0

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
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            use_amp=use_amp,
            scaler=scaler,
            history=history,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_path=best_path,
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
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            use_amp=use_amp,
            scaler=scaler,
            history=history,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_path=best_path,
        )

    final_path = _checkpoint_name(config.ANIMAL_TYPE, "final")
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "task": "binary_normal_abnormal",
            "animal_type": config.ANIMAL_TYPE,
            "best_val_loss": best_val_loss,
        },
        final_path,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(
        config.OUTPUT_DIR,
        f"training_history_{config.ANIMAL_TYPE}_binary_{ts}.json",
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ 이진 분류 학습 완료")
    print("=" * 60)
    print(f"  Best:  {best_path}")
    print(f"  Final: {final_path}")
    print(f"  History: {history_path}")


if __name__ == "__main__":
    train()

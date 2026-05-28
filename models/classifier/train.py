"""
EfficientNet-B3 멀티태스크 질환 분류 모델 학습 스크립트

실행 (프로젝트 루트에서):
  ANIMAL_TYPE=cat python models/classifier/train.py
  ANIMAL_TYPE=dog python models/classifier/train.py

Google Colab (CUDA, 300px, Focal + 고양이 WeightedRandomSampler):
  !ANIMAL_TYPE=cat python models/classifier/train.py

OOM 시 gradient accumulation:
  GRAD_ACCUM_STEPS=2 BATCH_SIZE=8 python models/classifier/train.py

맥북 (MPS):
  ANIMAL_TYPE=cat python models/classifier/train.py

CPU 강제:
  FORCE_CPU_TRAINING=1 python models/classifier/train.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import json

from models.classifier.model import create_model, count_parameters
from models.classifier.dataset import create_dataloader
from models.classifier.losses import build_per_disease_losses


class Config:
    # 데이터 - Training
    DOG_DATA_PATHS = [
        "eye_data/개/안구/일반",
        "eye_data/TL2/개/안구/일반",
    ]
    CAT_DATA_PATHS = [
        "eye_data/TL2/고양이/안구/일반",
    ]

    # 데이터 - Validation
    DOG_VAL_PATHS = [
        "eye_data/VL/개/안구/일반",
    ]
    CAT_VAL_PATHS = [
        "eye_data/VL/고양이/안구/일반",
    ]

    # 동물 (환경변수 ANIMAL_TYPE 으로 override)
    ANIMAL_TYPE = os.environ.get("ANIMAL_TYPE", "cat").strip().lower()

    # 입력 해상도 · 배치
    IMG_SIZE = 300
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))

    # 2단계 파인튜닝
    PHASE1_EPOCHS = int(os.environ.get("PHASE1_EPOCHS", "4"))   # 백본 freeze, 헤드만
    PHASE2_EPOCHS = int(os.environ.get("PHASE2_EPOCHS", "12"))  # 전체 unfreeze
    HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
    FINETUNE_LR = float(os.environ.get("FINETUNE_LR", "1e-5"))

    WEIGHT_DECAY = 1e-4
    HEAD_DROPOUT = 0.4

    # 손실: Focal Loss + Label Smoothing
    LOSS_TYPE = "focal"
    USE_CLASS_WEIGHTS = True
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1

    # 고양이만 WeightedRandomSampler
    USE_SAMPLER_FOR_CAT = True
    SAMPLER_BOOST_DISEASE = "결막염"
    SAMPLER_BOOST_FACTOR = 2.0

    # Early Stopping (검증 손실 기준)
    PATIENCE = 5

    # DataLoader
    NUM_WORKERS = 0  # 0 이면 device 기준 자동

    # 체크포인트
    OUTPUT_DIR = "models/classifier/checkpoints"
    RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "auto").strip()

    # Wandb
    USE_WANDB = False
    WANDB_PROJECT = "eye-disease-classification"


def get_device() -> str:
    """cuda → mps → cpu 순 자동 감지."""
    if os.environ.get("FORCE_CPU_TRAINING", "").strip().lower() in ("1", "true", "yes"):
        print("⚠ CPU 학습 모드 (FORCE_CPU_TRAINING=1)")
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA"
        print(f"✓ CUDA 사용 ({name})")
        return "cuda"
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) 사용")
        return "mps"
    print("⚠ CPU 사용 (학습이 느릴 수 있습니다)")
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


def _resolve_resume_path(config: Config) -> str:
    raw = config.RESUME_CHECKPOINT
    if not raw or raw.lower() in ("0", "false", "no", "off"):
        return ""
    if raw.lower() == "auto":
        path = os.path.join(config.OUTPUT_DIR, f"{config.ANIMAL_TYPE}_best.pth")
        return path if os.path.isfile(path) else ""
    return raw if os.path.isfile(raw) else ""


def _mean_disease_metric(metrics: Dict[str, float], suffix: str) -> float:
    vals = [v for k, v in metrics.items() if k.endswith(suffix)]
    return sum(vals) / len(vals) if vals else 0.0


def _init_disease_stats(diseases: List[str]) -> Tuple[dict, dict, dict, dict]:
    losses = {d: 0.0 for d in diseases}
    corrects = {d: 0 for d in diseases}
    totals = {d: 0 for d in diseases}
    recall_stats = {d: defaultdict(int) for d in diseases}  # per-class TP/FN
    return losses, corrects, totals, recall_stats


def _update_recall_stats(
    recall_stats: dict,
    disease: str,
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    for c in labels.unique().tolist():
        c = int(c)
        tp = ((preds == c) & (labels == c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        recall_stats[disease][f"tp_{c}"] += tp
        recall_stats[disease][f"fn_{c}"] += fn


def _compute_macro_recall(recall_stats: dict, disease: str) -> float:
    recalls = []
    classes = set()
    for k in recall_stats[disease]:
        if k.startswith("tp_"):
            classes.add(int(k.split("_")[1]))
    for c in sorted(classes):
        tp = recall_stats[disease].get(f"tp_{c}", 0)
        fn = recall_stats[disease].get(f"fn_{c}", 0)
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _run_forward_loss(
    model: nn.Module,
    criterion_dict: nn.ModuleDict,
    images: torch.Tensor,
    labels: dict,
    diseases: List[str],
    device: str,
) -> Tuple[torch.Tensor, dict, dict, dict, dict]:
    outputs = model(images)
    loss = torch.tensor(0.0, device=device)
    disease_losses = {}
    disease_corrects = {}
    disease_totals = {}
    recall_stats = {d: defaultdict(int) for d in diseases}

    for disease in diseases:
        disease_labels = labels[disease].to(device)
        disease_outputs = outputs[disease]
        valid_mask = disease_labels >= 0
        if valid_mask.sum() == 0:
            continue

        valid_labels = disease_labels[valid_mask]
        valid_outputs = disease_outputs[valid_mask]
        disease_loss = criterion_dict[disease](valid_outputs, valid_labels)
        loss = loss + disease_loss

        n = valid_mask.sum().item()
        disease_losses[disease] = disease_loss.item() * n
        _, preds = torch.max(valid_outputs, 1)
        disease_corrects[disease] = (preds == valid_labels).sum().item()
        disease_totals[disease] = n
        _update_recall_stats(recall_stats, disease, preds, valid_labels)

    return loss, disease_losses, disease_corrects, disease_totals, recall_stats


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_dict: nn.ModuleDict,
    optimizer: optim.Optimizer,
    device: str,
    diseases: List[str],
    *,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    d_losses, d_corrects, d_totals, d_recall = _init_disease_stats(diseases)
    merged_recall = {d: defaultdict(int) for d in diseases}

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, desc="Training")

    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)

        with autocast("cuda", enabled=use_amp):
            loss, batch_d_losses, batch_d_corrects, batch_d_totals, batch_recall = (
                _run_forward_loss(model, criterion_dict, images, labels, diseases, device)
            )
            loss = loss / grad_accum_steps

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

        for disease in diseases:
            if disease in batch_d_totals:
                d_losses[disease] += batch_d_losses[disease]
                d_corrects[disease] += batch_d_corrects[disease]
                d_totals[disease] += batch_d_totals[disease]
                for k, v in batch_recall[disease].items():
                    merged_recall[disease][k] += v

        progress.set_postfix({"loss": loss.item() * grad_accum_steps})

    return _build_metrics(total_loss, len(dataloader), d_losses, d_corrects, d_totals, merged_recall, diseases)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_dict: nn.ModuleDict,
    device: str,
    diseases: List[str],
    *,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    d_losses, d_corrects, d_totals, _ = _init_disease_stats(diseases)
    merged_recall = {d: defaultdict(int) for d in diseases}

    progress = tqdm(dataloader, desc="Validation")

    for images, labels in progress:
        images = images.to(device)
        with autocast("cuda", enabled=use_amp):
            loss, batch_d_losses, batch_d_corrects, batch_d_totals, batch_recall = (
                _run_forward_loss(model, criterion_dict, images, labels, diseases, device)
            )

        total_loss += loss.item()
        for disease in diseases:
            if disease in batch_d_totals:
                d_losses[disease] += batch_d_losses[disease]
                d_corrects[disease] += batch_d_corrects[disease]
                d_totals[disease] += batch_d_totals[disease]
                for k, v in batch_recall[disease].items():
                    merged_recall[disease][k] += v

    return _build_metrics(total_loss, len(dataloader), d_losses, d_corrects, d_totals, merged_recall, diseases)


def _build_metrics(
    total_loss: float,
    num_batches: int,
    d_losses: dict,
    d_corrects: dict,
    d_totals: dict,
    merged_recall: dict,
    diseases: List[str],
) -> Dict[str, float]:
    metrics = {"loss": total_loss / max(num_batches, 1)}
    for disease in diseases:
        if d_totals[disease] > 0:
            metrics[f"{disease}_loss"] = d_losses[disease] / d_totals[disease]
            metrics[f"{disease}_acc"] = d_corrects[disease] / d_totals[disease]
            metrics[f"{disease}_recall"] = _compute_macro_recall(merged_recall, disease)
    return metrics


def _print_disease_metrics(prefix: str, metrics: Dict[str, float], diseases: List[str]) -> None:
    print(f"\n{prefix} — 질환별 Accuracy / Recall:")
    for disease in diseases:
        acc_k = f"{disease}_acc"
        rec_k = f"{disease}_recall"
        if acc_k in metrics:
            print(
                f"  {disease:16s}  Acc: {metrics[acc_k]:.4f}  "
                f"Recall: {metrics.get(rec_k, 0.0):.4f}"
            )


def _build_optimizer(
    model: nn.Module,
    phase: int,
    config: Config,
) -> optim.Optimizer:
    if phase == 1:
        model.freeze_backbone()
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY)

    model.unfreeze_backbone()
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
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
    criterion_dict: nn.ModuleDict,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: str,
    diseases: List[str],
    config: Config,
    use_amp: bool,
    scaler: Optional[GradScaler],
    history: dict,
    best_val_loss: float,
    patience_counter: int,
    best_path: str,
) -> Tuple[float, int, str, int]:
    print(f"\n{'=' * 60}")
    print(f"Phase {phase}: {'헤드만 학습 (백본 freeze)' if phase == 1 else '전체 미세조정 (unfreeze)'}")
    print(f"  Epochs: {epochs}, LR: {config.HEAD_LR if phase == 1 else config.FINETUNE_LR}")
    print(f"{'=' * 60}")

    last_epoch = global_epoch_start

    for local_ep in range(1, epochs + 1):
        global_epoch = global_epoch_start + local_ep
        last_epoch = global_epoch
        print(f"\nEpoch {global_epoch} (Phase {phase} {local_ep}/{epochs})")
        print("-" * 60)

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion_dict,
            optimizer,
            device,
            diseases,
            use_amp=use_amp,
            scaler=scaler,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
        )
        val_metrics = validate_epoch(
            model,
            val_loader,
            criterion_dict,
            device,
            diseases,
            use_amp=use_amp,
        )
        scheduler.step()

        train_acc = _mean_disease_metric(train_metrics, "_acc")
        val_acc = _mean_disease_metric(val_metrics, "_acc")
        acc_gap = train_acc - val_acc

        print(f"\n[Train] Loss: {train_metrics['loss']:.4f}  Mean Acc: {train_acc:.4f}")
        print(f"[Val]   Loss: {val_metrics['loss']:.4f}  Mean Acc: {val_acc:.4f}")
        print(f"[Gap]   Train-Val Acc: {acc_gap:+.4f}  (과적합 모니터링)")

        _print_disease_metrics("[Val]", val_metrics, diseases)

        epoch_record = {
            "epoch": global_epoch,
            "phase": phase,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc_mean": train_acc,
            "val_acc_mean": val_acc,
            "acc_gap": acc_gap,
            "lr": optimizer.param_groups[0]["lr"],
            "val_per_disease": {
                d: {
                    "acc": val_metrics.get(f"{d}_acc"),
                    "recall": val_metrics.get(f"{d}_recall"),
                }
                for d in diseases
                if f"{d}_acc" in val_metrics
            },
        }
        history["epochs"].append(epoch_record)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc_mean"].append(train_acc)
        history["val_acc_mean"].append(val_acc)
        history["acc_gap"].append(acc_gap)

        if config.USE_WANDB:
            wandb.log({
                "epoch": global_epoch,
                "phase": phase,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_acc_mean": train_acc,
                "val_acc_mean": val_acc,
                "acc_gap": acc_gap,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_acc_mean": val_acc,
                    "config": {k: v for k, v in Config.__dict__.items() if not k.startswith("_")},
                },
                best_path,
            )
            print(f"✓ Best 모델 저장: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n⚠ Early Stopping (patience={config.PATIENCE}, val loss 기준)")
                return best_val_loss, patience_counter, best_path, last_epoch

    return best_val_loss, patience_counter, best_path, last_epoch


def train():
    config = Config()

    print("=" * 60)
    print("EfficientNet-B3 멀티태스크 질환 분류 모델 학습")
    print("=" * 60)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = get_device()
    batch_size = resolve_batch_size(config.BATCH_SIZE)
    num_workers = resolve_num_workers(device, config.NUM_WORKERS)
    use_amp = device == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    if config.ANIMAL_TYPE == "dog":
        train_paths = config.DOG_DATA_PATHS
        val_paths = config.DOG_VAL_PATHS
    else:
        train_paths = config.CAT_DATA_PATHS
        val_paths = config.CAT_VAL_PATHS

    use_sampler = config.ANIMAL_TYPE == "cat" and config.USE_SAMPLER_FOR_CAT
    boost_dis = config.SAMPLER_BOOST_DISEASE if use_sampler else None

    total_epochs = config.PHASE1_EPOCHS + config.PHASE2_EPOCHS

    print(f"\n⚙️  설정:")
    print(f"  - 동물: {config.ANIMAL_TYPE.upper()}")
    print(f"  - Phase1 (freeze): {config.PHASE1_EPOCHS} epochs, HEAD_LR={config.HEAD_LR}")
    print(f"  - Phase2 (unfreeze): {config.PHASE2_EPOCHS} epochs, LR={config.FINETUNE_LR}")
    print(f"  - Batch Size: {batch_size}  (accum={config.GRAD_ACCUM_STEPS})")
    print(f"  - Image Size: {config.IMG_SIZE}")
    print(f"  - Device: {device}  AMP: {'ON' if use_amp else 'OFF'}")
    print(f"  - Loss: Focal γ={config.FOCAL_GAMMA}, label_smoothing={config.LABEL_SMOOTHING}")
    print(f"  - WeightedSampler: {'ON (고양이)' if use_sampler else 'OFF (강아지)'}")
    if use_sampler:
        print(f"    · boost={boost_dis} ×{config.SAMPLER_BOOST_FACTOR}")

    pin_mem = device == "cuda"

    print(f"\n📊 데이터 로딩...")
    train_loader = create_dataloader(
        data_paths=train_paths,
        animal_type=config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        is_training=True,
        num_workers=num_workers,
        use_sampler=use_sampler,
        aug_preset="train",
        sampler_boost_disease=boost_dis,
        sampler_boost_factor=config.SAMPLER_BOOST_FACTOR,
        pin_memory=pin_mem,
    )
    val_loader = create_dataloader(
        data_paths=val_paths,
        animal_type=config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        is_training=False,
        num_workers=num_workers,
        use_sampler=False,
        aug_preset="default",
        pin_memory=pin_mem,
    )

    print(f"\n🔧 모델 생성...")
    model = create_model(
        animal_type=config.ANIMAL_TYPE,
        pretrained=True,
        head_dropout=config.HEAD_DROPOUT,
    ).to(device)

    resume_path = _resolve_resume_path(config)
    skip_phase1 = False
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        skip_phase1 = True
        print(f"✓ 체크포인트 로드: {resume_path} (epoch {ckpt.get('epoch', '?')})")
        print("  → Phase 1 건너뛰고 Phase 2 미세조정만 진행")

    print(f"학습 가능 파라미터: {count_parameters(model):,}")

    diseases = model.get_disease_names()
    criterion_dict = build_per_disease_losses(
        train_loader.dataset,
        diseases,
        config.LOSS_TYPE,
        device=device,
        use_class_weights=config.USE_CLASS_WEIGHTS,
        focal_gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING,
    ).to(device)

    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            config={
                "animal_type": config.ANIMAL_TYPE,
                "phase1_epochs": config.PHASE1_EPOCHS,
                "phase2_epochs": config.PHASE2_EPOCHS,
                "batch_size": batch_size,
                "img_size": config.IMG_SIZE,
            },
        )

    history = {
        "animal_type": config.ANIMAL_TYPE,
        "img_size": config.IMG_SIZE,
        "batch_size": batch_size,
        "grad_accum_steps": config.GRAD_ACCUM_STEPS,
        "loss_type": config.LOSS_TYPE,
        "focal_gamma": config.FOCAL_GAMMA,
        "label_smoothing": config.LABEL_SMOOTHING,
        "train_loss": [],
        "val_loss": [],
        "train_acc_mean": [],
        "val_acc_mean": [],
        "acc_gap": [],
        "epochs": [],
    }

    best_path = os.path.join(config.OUTPUT_DIR, f"{config.ANIMAL_TYPE}_best.pth")
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0

    print(f"\n🚀 학습 시작...\n")

    # Phase 1: backbone freeze
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
        )
        if patience_counter >= config.PATIENCE:
            pass  # early stop — still save final below
        else:
            patience_counter = 0  # phase2 에서 새 patience

    # Phase 2: full fine-tune
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
        )

    final_path = os.path.join(config.OUTPUT_DIR, f"{config.ANIMAL_TYPE}_final.pth")
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "config": {k: v for k, v in Config.__dict__.items() if not k.startswith("_")},
            "best_val_loss": best_val_loss,
        },
        final_path,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(
        config.OUTPUT_DIR,
        f"training_history_{config.ANIMAL_TYPE}_{ts}.json",
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"\n📂 저장 위치:")
    print(f"  - Best:  {best_path}")
    print(f"  - Final: {final_path}")
    print(f"  - History JSON: {history_path}")

    if config.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()

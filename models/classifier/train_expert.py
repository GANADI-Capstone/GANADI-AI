"""
질환 분류 전문가(Expert) 모델 학습 — 비정상 데이터만, N-class 단일 선택.

기본 (Mixup + Center Loss):
  ANIMAL_TYPE=cat python models/classifier/train_expert.py

SWA 포함:
  USE_SWA=1 ANIMAL_TYPE=cat python models/classifier/train_expert.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models.classifier.dataset_expert import create_expert_dataloader
from models.classifier.model_expert import FEAT_DIM, create_expert_model, count_parameters
from models.classifier.losses import FocalLoss


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


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

    USE_MIXUP = _env_flag("USE_MIXUP", "1")
    MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.2"))
    USE_CENTER_LOSS = _env_flag("USE_CENTER_LOSS", "1")
    CENTER_LOSS_LAMBDA = float(os.environ.get("CENTER_LOSS_LAMBDA", "0.01"))
    CENTER_LR = float(os.environ.get("CENTER_LR", "0.5"))
    USE_SWA = _env_flag("USE_SWA", "0")
    SWA_EPOCHS = int(os.environ.get("SWA_EPOCHS", "5"))

    FOCUS_DISEASE = os.environ.get("FOCUS_DISEASE", "결막염")

    PATIENCE = 5
    NUM_WORKERS = 0
    OUTPUT_DIR = "models/classifier/checkpoints"
    RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "auto").strip()


class CenterLoss(nn.Module):
    """클래스별 feature 중심점 — intra-class compact, inter-class separation."""

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return ((features - self.centers[labels]) ** 2).sum(dim=1).mean()


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def get_device() -> str:
    if os.environ.get("FORCE_CPU_TRAINING", "").strip().lower() in ("1", "true", "yes"):
        print("⚠ CPU 학습 모드")
        return "cpu"
    if torch.cuda.is_available():
        print(f"✓ CUDA ({torch.cuda.get_device_name(0)})")
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
    return os.path.join(Config.OUTPUT_DIR, f"{animal}_expert_{kind}.pth")


def _confusion_image_path(animal: str) -> str:
    return os.path.join(Config.OUTPUT_DIR, f"{animal}_expert_confusion_matrix.png")


def _resolve_resume_path(config: Config) -> str:
    raw = config.RESUME_CHECKPOINT
    if not raw or raw.lower() in ("0", "false", "no", "off"):
        return ""
    if raw.lower() == "auto":
        path = _checkpoint_name(config.ANIMAL_TYPE, "best")
        return path if os.path.isfile(path) else ""
    return raw if os.path.isfile(raw) else ""


def _compute_multiclass_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    num_classes: int,
    class_names: List[str],
) -> Dict:
    preds = preds.view(-1).cpu()
    labels = labels.view(-1).cpu()
    logits = logits.cpu()

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1

    total = labels.numel()
    acc = (preds == labels).sum().item() / total if total else 0.0

    if logits.numel() > 0 and logits.dim() == 2:
        top2 = logits.topk(min(2, num_classes), dim=1).indices
        top2_acc = (top2 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    else:
        top2_acc = 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []
    support_list: List[int] = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        support = int(cm[c, :].sum().item())
        name = class_names[c] if c < len(class_names) else str(c)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_list.append(f1)
        support_list.append(support)

    total_support = sum(support_list)
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    weighted_f1 = (
        sum(f * s for f, s in zip(f1_list, support_list)) / total_support
        if total_support
        else 0.0
    )

    return {
        "acc": acc,
        "top2_acc": top2_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def _print_confusion_matrix(cm: List[List[int]], class_names: List[str]) -> None:
    print("\nConfusion Matrix (행=실제, 열=예측):")
    header = "          " + " ".join(f"{name[:6]:>7s}" for name in class_names)
    print(header)
    for i, row in enumerate(cm):
        name = class_names[i][:6] if i < len(class_names) else str(i)
        vals = " ".join(f"{v:7d}" for v in row)
        print(f"{name:>8s}  {vals}")


def _print_per_class_metrics(per_class: Dict[str, Dict[str, float]]) -> None:
    print("\n질환별 Recall / Precision / F1:")
    for name, m in per_class.items():
        print(
            f"  {name:16s}  P: {m['precision']:.4f}  "
            f"R: {m['recall']:.4f}  F1: {m['f1']:.4f}  (n={m['support']})"
        )


def _save_confusion_heatmap(
    cm: List[List[int]],
    class_names: List[str],
    save_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Expert Disease Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✓ Confusion heatmap 저장: {save_path}")
    except ImportError:
        print("⚠ matplotlib/seaborn 미설치 — heatmap 생략 (pip install matplotlib seaborn)")


def _compute_batch_loss(
    logits: torch.Tensor,
    features: Optional[torch.Tensor],
    labels_a: torch.Tensor,
    labels_b: Optional[torch.Tensor],
    lam: float,
    criterion: nn.Module,
    center_loss: Optional[CenterLoss],
    center_lambda: float,
) -> torch.Tensor:
    if labels_b is not None and lam != 1.0:
        loss_ce = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(
            logits, labels_b
        )
        if center_loss is not None and features is not None:
            loss_c = lam * center_loss(features, labels_a) + (1.0 - lam) * center_loss(
                features, labels_b
            )
            return loss_ce + center_lambda * loss_c
        return loss_ce

    loss_ce = criterion(logits, labels_a)
    if center_loss is not None and features is not None:
        return loss_ce + center_lambda * center_loss(features, labels_a)
    return loss_ce


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    class_names: List[str],
    *,
    optimizer: Optional[optim.Optimizer] = None,
    center_loss: Optional[CenterLoss] = None,
    center_optimizer: Optional[optim.Optimizer] = None,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    grad_accum_steps: int = 1,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    use_center_loss: bool = False,
    center_lambda: float = 0.01,
) -> Tuple[float, Dict]:
    is_train = optimizer is not None
    model.train(is_train)
    if center_loss is not None:
        center_loss.train(is_train)

    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []

    progress = tqdm(dataloader, desc="Training" if is_train else "Validation")

    if is_train:
        optimizer.zero_grad(set_to_none=True)
        if center_optimizer is not None:
            center_optimizer.zero_grad(set_to_none=True)

    need_features = is_train and use_center_loss

    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        labels_a, labels_b, lam = labels, None, 1.0
        if is_train and use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

        if is_train:
            with autocast("cuda", enabled=use_amp):
                if need_features:
                    features, logits = model(images, return_features=True)
                else:
                    features, logits = None, model(images)
                loss = _compute_batch_loss(
                    logits,
                    features,
                    labels_a,
                    labels_b,
                    lam,
                    criterion,
                    center_loss if use_center_loss else None,
                    center_lambda,
                ) / grad_accum_steps

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
                if center_optimizer is not None:
                    center_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if center_optimizer is not None:
                    center_optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum_steps
        else:
            with torch.no_grad(), autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_logits.append(logits.float().cpu())
        progress.set_postfix({"loss": loss.item()})

    metrics = _compute_multiclass_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_logits),
        num_classes,
        class_names,
    )
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics["loss"], metrics


def _build_optimizer(model: nn.Module, phase: int, config: Config) -> optim.Optimizer:
    if phase == 1:
        model.freeze_backbone()
        params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY)

    model.unfreeze_backbone()
    head_params = [
        p for name, p in model.named_parameters() if not name.startswith("backbone.")
    ]
    return optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.FINETUNE_LR},
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
    center_loss: Optional[CenterLoss],
    center_optimizer: Optional[optim.Optimizer],
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: str,
    config: Config,
    num_classes: int,
    class_names: List[str],
    use_amp: bool,
    scaler: Optional[GradScaler],
    history: dict,
    best_val_loss: float,
    patience_counter: int,
    best_path: str,
    swa_model: Optional[AveragedModel] = None,
) -> Tuple[float, int, str, int]:
    print(f"\n{'=' * 60}")
    print(f"Phase {phase}: {'헤드만 (freeze)' if phase == 1 else '전체 미세조정'}")
    print(f"{'=' * 60}")

    last_epoch = global_epoch_start
    swa_start_local = max(1, epochs - config.SWA_EPOCHS + 1)

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
            num_classes,
            class_names,
            optimizer=optimizer,
            center_loss=center_loss,
            center_optimizer=center_optimizer,
            use_amp=use_amp,
            scaler=scaler,
            grad_accum_steps=config.GRAD_ACCUM_STEPS,
            use_mixup=config.USE_MIXUP,
            mixup_alpha=config.MIXUP_ALPHA,
            use_center_loss=config.USE_CENTER_LOSS,
            center_lambda=config.CENTER_LOSS_LAMBDA,
        )
        _, val_m = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            class_names,
            use_amp=use_amp,
        )
        scheduler.step()

        if (
            config.USE_SWA
            and swa_model is not None
            and phase == 2
            and local_ep >= swa_start_local
        ):
            swa_model.update_parameters(model)
            print(f"  [SWA] 가중치 평균 업데이트 (Phase2 {local_ep}/{epochs})")

        acc_gap = train_m["acc"] - val_m["acc"]
        focus = config.FOCUS_DISEASE
        focus_m = val_m["per_class"].get(focus, {})

        print(
            f"[Train] Loss: {train_m['loss']:.4f}  Acc: {train_m['acc']:.4f}  "
            f"Macro-F1: {train_m['macro_f1']:.4f}"
        )
        print(
            f"[Val]   Loss: {val_m['loss']:.4f}  Acc: {val_m['acc']:.4f}  "
            f"Top-2: {val_m['top2_acc']:.4f}  Macro-F1: {val_m['macro_f1']:.4f}  "
            f"Weighted-F1: {val_m['weighted_f1']:.4f}"
        )
        print(f"[Gap]   Train-Val Acc: {acc_gap:+.4f}")
        if focus_m:
            print(
                f"[Focus] {focus} — Val R: {focus_m.get('recall', 0):.4f}  "
                f"P: {focus_m.get('precision', 0):.4f}  F1: {focus_m.get('f1', 0):.4f}"
            )

        _print_per_class_metrics(val_m["per_class"])
        _print_confusion_matrix(val_m["confusion_matrix"], class_names)

        record = {
            "epoch": global_epoch,
            "phase": phase,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "train_acc": train_m["acc"],
            "val_acc": val_m["acc"],
            "val_top2_acc": val_m["top2_acc"],
            "train_macro_f1": train_m["macro_f1"],
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            "acc_gap": acc_gap,
            "val_per_class": val_m["per_class"],
            "val_confusion_matrix": val_m["confusion_matrix"],
            f"val_{focus}_recall": focus_m.get("recall"),
            f"val_{focus}_f1": focus_m.get("f1"),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history["epochs"].append(record)
        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["acc"])
        history["val_acc"].append(val_m["acc"])
        history["acc_gap"].append(acc_gap)
        history.setdefault("val_top2_acc", []).append(val_m["top2_acc"])
        history.setdefault("val_macro_f1", []).append(val_m["macro_f1"])
        history.setdefault("focus_disease_recall", []).append(focus_m.get("recall", 0.0))

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
                    "val_top2_acc": val_m["top2_acc"],
                    "val_macro_f1": val_m["macro_f1"],
                    "val_per_class": val_m["per_class"],
                    "task": "expert_disease_classifier",
                    "animal_type": config.ANIMAL_TYPE,
                    "class_names": class_names,
                    "num_classes": num_classes,
                },
                best_path,
            )
            _save_confusion_heatmap(
                val_m["confusion_matrix"],
                class_names,
                _confusion_image_path(config.ANIMAL_TYPE),
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
    print("질환 분류 Expert 모델 학습 (비정상 데이터만)")
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
    print(
        f"  Mixup: {'ON' if config.USE_MIXUP else 'OFF'} (α={config.MIXUP_ALPHA})  "
        f"CenterLoss: {'ON' if config.USE_CENTER_LOSS else 'OFF'} (λ={config.CENTER_LOSS_LAMBDA})  "
        f"SWA: {'ON' if config.USE_SWA else 'OFF'}"
    )

    pin_mem = device == "cuda"
    train_loader = create_expert_dataloader(
        train_paths,
        config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        is_training=True,
        num_workers=num_workers,
        use_sampler=config.USE_SAMPLER,
        pin_memory=pin_mem,
    )
    val_loader = create_expert_dataloader(
        val_paths,
        config.ANIMAL_TYPE,
        batch_size=batch_size,
        img_size=config.IMG_SIZE,
        is_training=False,
        num_workers=num_workers,
        use_sampler=False,
        pin_memory=pin_mem,
    )

    model = create_expert_model(
        animal_type=config.ANIMAL_TYPE,
        head_dropout=config.HEAD_DROPOUT,
    ).to(device)

    num_classes = model.num_classes
    class_names = model.class_names

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

    center_loss: Optional[CenterLoss] = None
    center_optimizer: Optional[optim.Optimizer] = None
    if config.USE_CENTER_LOSS:
        center_loss = CenterLoss(num_classes, FEAT_DIM).to(device)
        center_optimizer = optim.SGD(center_loss.parameters(), lr=config.CENTER_LR)
        print(f"✓ Center Loss 활성 (feat_dim={FEAT_DIM}, center_lr={config.CENTER_LR})")

    swa_model: Optional[AveragedModel] = None
    if config.USE_SWA:
        swa_model = AveragedModel(model)
        print(f"✓ SWA 활성 (Phase2 마지막 {config.SWA_EPOCHS} epoch 평균)")

    history = {
        "task": "expert_disease_classifier",
        "animal_type": config.ANIMAL_TYPE,
        "num_classes": num_classes,
        "class_names": class_names,
        "use_mixup": config.USE_MIXUP,
        "use_center_loss": config.USE_CENTER_LOSS,
        "use_swa": config.USE_SWA,
        "focus_disease": config.FOCUS_DISEASE,
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "acc_gap": [],
        "val_top2_acc": [],
        "val_macro_f1": [],
        "focus_disease_recall": [],
        "epochs": [],
    }

    best_path = _checkpoint_name(config.ANIMAL_TYPE, "best")
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0

    phase_kwargs = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        center_loss=center_loss,
        center_optimizer=center_optimizer,
        device=device,
        config=config,
        num_classes=num_classes,
        class_names=class_names,
        use_amp=use_amp,
        scaler=scaler,
        history=history,
        swa_model=swa_model,
    )

    if not skip_phase1 and config.PHASE1_EPOCHS > 0:
        optimizer = _build_optimizer(model, phase=1, config=config)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.PHASE1_EPOCHS)
        best_val_loss, patience_counter, best_path, last_epoch = _run_phase(
            phase=1,
            epochs=config.PHASE1_EPOCHS,
            global_epoch_start=0,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_path=best_path,
            **phase_kwargs,
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
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            best_path=best_path,
            **phase_kwargs,
        )

    final_path = _checkpoint_name(config.ANIMAL_TYPE, "final")
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "task": "expert_disease_classifier",
            "animal_type": config.ANIMAL_TYPE,
            "class_names": class_names,
            "num_classes": num_classes,
            "best_val_loss": best_val_loss,
        },
        final_path,
    )

    if config.USE_SWA and swa_model is not None:
        print("\n[SWA] BatchNorm 통계 갱신 중...")
        update_bn(train_loader, swa_model, device=device)
        swa_path = _checkpoint_name(config.ANIMAL_TYPE, "swa")
        torch.save(
            {
                "epoch": last_epoch,
                "model_state_dict": swa_model.module.state_dict(),
                "task": "expert_disease_classifier",
                "animal_type": config.ANIMAL_TYPE,
                "class_names": class_names,
                "num_classes": num_classes,
                "swa": True,
            },
            swa_path,
        )
        print(f"✓ SWA 모델 저장: {swa_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(
        config.OUTPUT_DIR,
        f"training_history_{config.ANIMAL_TYPE}_expert_{ts}.json",
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ Expert 학습 완료")
    print("=" * 60)
    print(f"  Best:  {best_path}")
    print(f"  Final: {final_path}")
    if config.USE_SWA:
        print(f"  SWA:   {_checkpoint_name(config.ANIMAL_TYPE, 'swa')}")
    print(f"  History: {history_path}")
    print(f"  CM img:  {_confusion_image_path(config.ANIMAL_TYPE)}")


if __name__ == "__main__":
    train()

"""공통 학습 유틸: EMA, Mixup/CutMix, 메트릭, 루프."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.utils import ModelEmaV2
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

EMA_DECAY = 0.9998


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def create_ema(model: nn.Module, device: str, decay: float = EMA_DECAY) -> ModelEmaV2:
    ema = ModelEmaV2(model, decay=decay, device=device)
    print(f"✓ EMA 활성 (decay={decay})")
    return ema


def ema_state_dict(ema: Optional[ModelEmaV2], model: nn.Module) -> dict:
    if ema is not None:
        return ema.module.state_dict()
    return model.state_dict()


def eval_model(model: nn.Module, ema: Optional[ModelEmaV2]) -> nn.Module:
    return ema.module if ema is not None else model


def get_device() -> str:
    if os.environ.get("FORCE_CPU_TRAINING", "").strip().lower() in ("1", "true", "yes"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_batch_size(configured: int) -> int:
    env_bs = os.environ.get("BATCH_SIZE", "").strip()
    if env_bs.isdigit() and int(env_bs) > 0:
        return int(env_bs)
    return configured or 16


def resolve_num_workers(device: str, default: int = 0) -> int:
    env_nw = os.environ.get("NUM_WORKERS", "").strip()
    if env_nw.isdigit():
        return int(env_nw)
    return 2 if device == "cuda" else default


def resolve_img_size(default: int = 300) -> int:
    env = os.environ.get("IMG_SIZE", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return default


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size, _, height, width = x.size()
    index = torch.randperm(batch_size, device=x.device)

    cut_ratio = np.sqrt(max(1.0 - lam, 0.0))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, y, y[index], float(lam)


def apply_mix(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    mix_choice_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float, str]:
    """Mixup/CutMix 적용. mix_choice_prob: CutMix 선택 확률 (둘 다 켜진 경우)."""
    if use_cutmix and use_mixup:
        if np.random.rand() < mix_choice_prob:
            x, la, lb, lam = cutmix_data(images, labels, cutmix_alpha)
            return x, la, lb, lam, "cutmix"
        x, la, lb, lam = mixup_data(images, labels, mixup_alpha)
        return x, la, lb, lam, "mixup"
    if use_cutmix:
        x, la, lb, lam = cutmix_data(images, labels, cutmix_alpha)
        return x, la, lb, lam, "cutmix"
    if use_mixup:
        x, la, lb, lam = mixup_data(images, labels, mixup_alpha)
        return x, la, lb, lam, "mixup"
    return images, labels, None, 1.0, "none"


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    class_names: List[str],
) -> Dict[str, Any]:
    num_classes = len(class_names)
    preds, labels = preds.view(-1).cpu(), labels.view(-1).cpu()
    logits = logits.cpu()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    total = max(labels.numel(), 1)
    acc = (preds == labels).sum().item() / total
    top2 = logits.topk(min(2, num_classes), dim=1).indices
    top2_acc = (top2 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    per_class: Dict[str, Dict[str, float]] = {}
    f1s, weights = [], []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        sup = int(cm[c, :].sum().item())
        name = class_names[c]
        per_class[name] = {"precision": p, "recall": r, "f1": f1, "support": sup}
        f1s.append(f1)
        weights.append(sup)
    wsum = sum(weights) or 1
    return {
        "acc": acc,
        "top2_acc": top2_acc,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "weighted_f1": sum(f * w for f, w in zip(f1s, weights)) / wsum,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def print_confusion(cm: List[List[int]], names: List[str]) -> None:
    print("\nConfusion Matrix (행=실제, 열=예측):")
    hdr = "          " + " ".join(f"{n[:6]:>7s}" for n in names)
    print(hdr)
    for i, row in enumerate(cm):
        print(f"{names[i][:6]:>8s}  " + " ".join(f"{v:7d}" for v in row))


def print_per_class(per_class: Dict) -> None:
    print("\n질환별 P / R / F1:")
    for name, m in per_class.items():
        print(
            f"  {name:10s}  P:{m['precision']:.3f}  R:{m['recall']:.3f}  "
            f"F1:{m['f1']:.3f}  n={m['support']}"
        )


def save_confusion_heatmap(cm, names, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"✓ Confusion heatmap: {path}")
    except ImportError:
        print("⚠ matplotlib/seaborn 없음 — heatmap 생략")


def _forward_logits(model, images, need_projection: bool):
    if need_projection:
        out = model(images, return_projection=True)
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None
    return model(images), None


def run_epoch(
    model,
    loader,
    criterion,
    device,
    class_names,
    optimizer=None,
    use_amp=False,
    scaler=None,
    grad_accum=1,
    use_mixup=False,
    mixup_alpha=0.2,
    use_cutmix=False,
    cutmix_alpha=1.0,
    mix_choice_prob=0.5,
    ema: Optional[ModelEmaV2] = None,
    eval_ema: Optional[ModelEmaV2] = None,
    supcon_criterion=None,
    supcon_weight: float = 0.5,
    ce_weight: float = 0.5,
) -> Tuple[float, Dict]:
    train = optimizer is not None
    infer_model = eval_model(model, eval_ema if not train else None)
    infer_model.train(train)
    total_loss = 0.0
    preds_all, labels_all, logits_all = [], [], []
    if train:
        optimizer.zero_grad(set_to_none=True)

    need_projection = supcon_criterion is not None and train

    for step, (images, labels) in enumerate(
        tqdm(loader, desc="Train" if train else "Val"), 1,
    ):
        images, labels = images.to(device), labels.to(device)
        la, lb, lam = labels, None, 1.0
        mix_mode = "none"
        if train and (use_mixup or use_cutmix):
            images, la, lb, lam, mix_mode = apply_mix(
                images, labels,
                use_mixup=use_mixup,
                mixup_alpha=mixup_alpha,
                use_cutmix=use_cutmix,
                cutmix_alpha=cutmix_alpha,
                mix_choice_prob=mix_choice_prob,
            )

        if train:
            with autocast("cuda", enabled=use_amp):
                logits, projections = _forward_logits(model, images, need_projection)
                if lb is not None:
                    ce = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
                else:
                    ce = criterion(logits, la)

                if supcon_criterion is not None and projections is not None and mix_mode == "none":
                    loss = ce_weight * ce + supcon_weight * supcon_criterion(projections, labels)
                else:
                    loss = ce
                loss = loss / grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % grad_accum == 0 or step == len(loader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            total_loss += loss.item() * grad_accum
        else:
            with torch.no_grad(), autocast("cuda", enabled=use_amp):
                logits, _ = _forward_logits(infer_model, images, False)
                loss = criterion(logits, labels)
            total_loss += loss.item()

        _, preds = logits.max(1)
        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())
        logits_all.append(logits.float().cpu())

    metrics = compute_metrics(
        torch.cat(preds_all), torch.cat(labels_all), torch.cat(logits_all), class_names,
    )
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics["loss"], metrics


def build_optimizer(model, phase: int, cfg) -> optim.AdamW:
    if phase == 1:
        model.freeze_backbone()
        return optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.HEAD_LR, weight_decay=cfg.WEIGHT_DECAY,
        )
    model.unfreeze_backbone()
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": cfg.FINETUNE_LR},
         {"params": head, "lr": cfg.FINETUNE_LR}],
        weight_decay=cfg.WEIGHT_DECAY,
    )


def build_scheduler(optimizer, epochs: int, scheduler_type: str = "cosine"):
    if scheduler_type == "warm_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def run_training_loop(
    model, train_loader, val_loader, criterion, cfg, device,
    class_names, best_path, cm_path, history_path_prefix,
    use_amp, scaler, resume_skip_phase1=False,
    task: str = "group_classifier",
    extra_ckpt: Optional[dict] = None,
    ema: Optional[ModelEmaV2] = None,
    supcon_criterion=None,
    supcon_weight: float = 0.5,
    ce_weight: float = 0.5,
    scheduler_type: str = "cosine",
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    mix_choice_prob: float = 0.5,
) -> str:
    history = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": [],
    }
    best_val = float("inf")
    patience = 0
    last_ep = 0
    stopped = False

    use_mixup = getattr(cfg, "USE_MIXUP", False)
    mixup_alpha = getattr(cfg, "MIXUP_ALPHA", 0.2)

    def one_phase(phase, epochs, start_ep):
        nonlocal best_val, patience, last_ep, stopped
        opt = build_optimizer(model, phase, cfg)
        sched = build_scheduler(opt, epochs, scheduler_type)
        print(f"\nPhase {phase} ({epochs} epochs, scheduler={scheduler_type})")
        for local in range(1, epochs + 1):
            ep = start_ep + local
            last_ep = ep
            print(f"\nEpoch {ep} (Phase {phase} {local}/{epochs})")
            _, tr = run_epoch(
                model, train_loader, criterion, device, class_names,
                optimizer=opt, use_amp=use_amp, scaler=scaler,
                grad_accum=cfg.GRAD_ACCUM_STEPS,
                use_mixup=use_mixup, mixup_alpha=mixup_alpha,
                use_cutmix=use_cutmix, cutmix_alpha=cutmix_alpha,
                mix_choice_prob=mix_choice_prob,
                ema=ema, supcon_criterion=supcon_criterion,
                supcon_weight=supcon_weight, ce_weight=ce_weight,
            )
            _, va = run_epoch(
                model, val_loader, criterion, device, class_names,
                use_amp=use_amp, eval_ema=ema,
            )
            sched.step()
            print(
                f"[Train] loss={tr['loss']:.4f} acc={tr['acc']:.4f}  "
                f"[Val] loss={va['loss']:.4f} acc={va['acc']:.4f} "
                f"macro_f1={va['macro_f1']:.4f} top2={va['top2_acc']:.4f}"
            )
            print_per_class(va["per_class"])
            print_confusion(va["confusion_matrix"], class_names)
            history["epochs"].append({"epoch": ep, "phase": phase, "val": va})
            history["train_loss"].append(tr["loss"])
            history["val_loss"].append(va["loss"])
            history["val_acc"].append(va["acc"])
            history["val_macro_f1"].append(va["macro_f1"])

            if va["loss"] < best_val:
                best_val = va["loss"]
                patience = 0
                payload = {
                    "epoch": ep,
                    "model_state_dict": ema_state_dict(ema, model),
                    "val_loss": va["loss"],
                    "val_acc": va["acc"],
                    "val_per_class": va["per_class"],
                    "task": task,
                    "class_names": class_names,
                    "ema_decay": EMA_DECAY if ema else None,
                }
                if extra_ckpt:
                    payload.update(extra_ckpt)
                torch.save(payload, best_path)
                save_confusion_heatmap(va["confusion_matrix"], class_names, cm_path)
                tag = " (EMA weights)" if ema else ""
                print(f"✓ Best: {best_path}{tag}")
            else:
                patience += 1
                if patience >= cfg.PATIENCE:
                    print(f"Early stop (patience={cfg.PATIENCE})")
                    stopped = True
                    return

    if not resume_skip_phase1 and cfg.PHASE1_EPOCHS > 0:
        one_phase(1, cfg.PHASE1_EPOCHS, 0)
    if not stopped and cfg.PHASE2_EPOCHS > 0:
        patience = 0
        one_phase(2, cfg.PHASE2_EPOCHS, last_ep)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = f"{history_path_prefix}_{ts}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"History: {hist_path}")
    return hist_path

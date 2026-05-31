"""
멀티태스크 헤드 불균형 해소 — 백본 freeze + 헤드만 재학습.

실행:
  ANIMAL_TYPE=dog python models/classifier/train_head_rebalance.py --epochs 8 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from timm.utils import ModelEmaV2
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    evaluate_multitask_topk,
)
from models.classifier.inference_multitask import extract_state_dict
from models.classifier.losses import FocalLoss
from models.classifier.model import create_model
from models.classifier.train import (
    Config as BaseTrainConfig,
    _build_metrics,
    _init_disease_stats,
    _mean_disease_metric,
    _print_disease_metrics,
    _update_recall_stats,
    get_device,
    resolve_batch_size,
    resolve_num_workers,
)
from models.classifier.train_common import create_ema, ema_state_dict, eval_model
from models.classifier.train_random_split import RandomSplitConfig

DEFAULT_DOG_HEAD_LOSS_WEIGHTS: Dict[str, float] = {
    "백내장": 0.5,
    "핵경화": 1.5,
    "결막염": 1.5,
    "색소침착성각막염": 1.5,
    "안검염": 1.5,
    "안검종양": 1.5,
}

FOCAL_GAMMA = 2.0
EFFECTIVE_BETA = 0.999
LABEL_SMOOTHING = 0.1


def parse_head_weights(spec: Optional[str], diseases: Sequence[str]) -> Dict[str, float]:
    weights = {d: 1.0 for d in diseases}
    if not spec:
        for d in diseases:
            if d in DEFAULT_DOG_HEAD_LOSS_WEIGHTS:
                weights[d] = DEFAULT_DOG_HEAD_LOSS_WEIGHTS[d]
        return weights
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, val = part.split(":", 1)
        name = name.strip()
        if name not in diseases:
            raise ValueError(f"알 수 없는 질환 '{name}'")
        weights[name] = float(val)
    return weights


def effective_number_class_weights(counts: List[int], beta: float, device: str) -> torch.Tensor:
    weights = []
    for n in counts:
        if n <= 0:
            weights.append(0.0)
        else:
            eff = (1.0 - beta ** n) / (1.0 - beta)
            weights.append(1.0 / max(eff, 1e-8))
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    if w.sum() > 0:
        w = w / w.sum() * len(counts)
    else:
        w = torch.ones(len(counts), device=device)
    return w


def count_disease_class_samples(train_ds: RandomSplitEyeDataset, disease: str, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for idx in train_ds.indices:
        _, label_dict = train_ds.base.samples[idx]
        lv = label_dict[disease]
        if 0 <= lv < num_classes:
            counts[lv] += 1
    return counts


def build_rebalance_losses(train_ds: RandomSplitEyeDataset, diseases: List[str], device: str) -> nn.ModuleDict:
    modules: Dict[str, nn.Module] = {}
    print("\n📐 Focal + effective-number class weights:")
    for disease in diseases:
        num_classes = len(train_ds.label_map[disease])
        counts = count_disease_class_samples(train_ds, disease, num_classes)
        alpha = effective_number_class_weights(counts, EFFECTIVE_BETA, device)
        modules[disease] = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha, label_smoothing=LABEL_SMOOTHING)
        print(f"  {disease:16s} n={counts}")
    return nn.ModuleDict(modules)


def _run_forward_loss_weighted(
    model: nn.Module,
    criterion_dict: nn.ModuleDict,
    images: torch.Tensor,
    labels: dict,
    diseases: List[str],
    device: str,
    head_loss_weights: Dict[str, float],
) -> Tuple[torch.Tensor, dict, dict, dict, dict]:
    outputs = model(images)
    loss = torch.tensor(0.0, device=device)
    disease_losses: Dict[str, float] = {}
    disease_corrects: Dict[str, int] = {}
    disease_totals: Dict[str, int] = {}
    recall_stats = {d: defaultdict(int) for d in diseases}

    for disease in diseases:
        disease_labels = labels[disease].to(device)
        valid_mask = disease_labels >= 0
        if valid_mask.sum() == 0:
            continue
        valid_labels = disease_labels[valid_mask]
        valid_outputs = outputs[disease][valid_mask]
        dloss = criterion_dict[disease](valid_outputs, valid_labels)
        hw = head_loss_weights.get(disease, 1.0)
        weighted = dloss * hw
        loss = loss + weighted
        n = int(valid_mask.sum().item())
        disease_losses[disease] = weighted.item() * n
        _, preds = torch.max(valid_outputs, 1)
        disease_corrects[disease] = int((preds == valid_labels).sum().item())
        disease_totals[disease] = n
        _update_recall_stats(recall_stats, disease, preds, valid_labels)

    return loss, disease_losses, disease_corrects, disease_totals, recall_stats


def set_backbone_frozen(model: nn.Module) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()


def unfreeze_backbone_last_blocks(model: nn.Module, n_blocks: int) -> int:
    """마지막 n_blocks만 unfreeze. frozen block 인덱스 상한(start) 반환."""
    blocks = model.backbone.blocks
    total = len(blocks)
    start = max(0, total - n_blocks)
    for i in range(total):
        block = blocks[i]
        trainable = i >= start
        for p in block.parameters():
            p.requires_grad = trainable
        block.train() if trainable else block.eval()
    print(f"  Unfroze backbone blocks [{start}:{total})")
    return start


def apply_backbone_bn_modes(
    model: nn.Module,
    *,
    freeze_backbone: bool,
    unfreeze_block_start: Optional[int] = None,
) -> None:
    """BatchNorm running stats: frozen 구간은 eval, unfreeze 구간만 train."""
    if freeze_backbone:
        model.backbone.eval()
        return
    if unfreeze_block_start is None:
        return
    for i, block in enumerate(model.backbone.blocks):
        block.train() if i >= unfreeze_block_start else block.eval()
    for name, module in model.backbone.named_modules():
        if name.startswith("blocks."):
            continue
        module.eval()


def head_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.startswith("classifiers.") and p.requires_grad]


def build_optimizer(
    model: nn.Module,
    head_lr: float,
    backbone_lr: Optional[float],
    unfreeze_blocks: int,
    freeze_backbone: bool,
) -> optim.Optimizer:
    wd = BaseTrainConfig.WEIGHT_DECAY
    if unfreeze_blocks > 0:
        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        hd = head_parameters(model)
        return optim.AdamW(
            [{"params": bb, "lr": backbone_lr}, {"params": hd, "lr": head_lr}],
            weight_decay=wd,
        )
    if freeze_backbone:
        set_backbone_frozen(model)
        return optim.AdamW(head_parameters(model), lr=head_lr, weight_decay=wd)
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    return optim.AdamW(model.parameters(), lr=head_lr, weight_decay=wd)


def train_epoch_rebalance(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_dict: nn.ModuleDict,
    optimizer: optim.Optimizer,
    device: str,
    diseases: List[str],
    head_loss_weights: Dict[str, float],
    freeze_backbone: bool,
    unfreeze_block_start: Optional[int],
    use_amp: bool,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEmaV2],
) -> Dict[str, float]:
    model.train()
    apply_backbone_bn_modes(
        model, freeze_backbone=freeze_backbone, unfreeze_block_start=unfreeze_block_start,
    )
    total_loss = 0.0
    d_losses, d_corrects, d_totals, d_recall = _init_disease_stats(diseases)
    merged_recall = {d: defaultdict(int) for d in diseases}
    optimizer.zero_grad(set_to_none=True)
    for images, labels in tqdm(dataloader, desc="Train"):
        images = images.to(device)
        with autocast("cuda", enabled=use_amp):
            loss, b_dl, b_dc, b_dt, b_rr = _run_forward_loss_weighted(
                model, criterion_dict, images, labels, diseases, device, head_loss_weights,
            )
        if use_amp and scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema:
            ema.update(model)
        total_loss += loss.item()
        for d in diseases:
            if d in b_dt:
                d_losses[d] += b_dl[d]
                d_corrects[d] += b_dc[d]
                d_totals[d] += b_dt[d]
                for k, v in b_rr[d].items():
                    merged_recall[d][k] += v
    return _build_metrics(total_loss, len(dataloader), d_losses, d_corrects, d_totals, merged_recall, diseases)


@torch.no_grad()
def validate_epoch_rebalance(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_dict: nn.ModuleDict,
    device: str,
    diseases: List[str],
    head_loss_weights: Dict[str, float],
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    d_losses, d_corrects, d_totals, _ = _init_disease_stats(diseases)
    merged_recall = {d: defaultdict(int) for d in diseases}
    for images, labels in tqdm(dataloader, desc="Val"):
        images = images.to(device)
        with autocast("cuda", enabled=use_amp):
            loss, b_dl, b_dc, b_dt, b_rr = _run_forward_loss_weighted(
                model, criterion_dict, images, labels, diseases, device, head_loss_weights,
            )
        total_loss += loss.item()
        for d in diseases:
            if d in b_dt:
                d_losses[d] += b_dl[d]
                d_corrects[d] += b_dc[d]
                d_totals[d] += b_dt[d]
                for k, v in b_rr[d].items():
                    merged_recall[d][k] += v
    return _build_metrics(total_loss, len(dataloader), d_losses, d_corrects, d_totals, merged_recall, diseases)


@torch.no_grad()
def quick_top1_confusion_summary(
    model: nn.Module,
    val_ds: RandomSplitEyeDataset,
    val_loader: DataLoader,
    diseases: List[str],
    device: str,
    focus_pred: Optional[str] = "백내장",
) -> Dict[str, Dict[str, int]]:
    model.eval()
    cm = {d: {p: 0 for p in diseases} for d in diseases}
    local_idx = 0
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        bs = images.size(0)
        for i in range(bs):
            if local_idx >= len(val_ds):
                break
            gt_d, gt_y = _active_disease_and_label(labels, i, diseases)
            if gt_d is None or gt_y <= 0:
                local_idx += 1
                continue
            pred_top1 = _rank_diseases_by_abnormal_prob(outputs, i, diseases)[0][0]
            cm[gt_d][pred_top1] += 1
            local_idx += 1

    print(f"\n{'=' * 64}\nTop-1 Confusion (Val 비정상)\n{'=' * 64}")
    short = [d[:6] for d in diseases]
    print("          " + " ".join(f"{s:>7s}" for s in short))
    for gt in diseases:
        row = cm[gt]
        total = sum(row.values())
        if not total:
            continue
        print(f"{gt[:8]:>8s}  " + " ".join(f"{row[p]:7d}" for p in diseases))
        if focus_pred and focus_pred in diseases:
            mis = row.get(focus_pred, 0)
            print(f"           → {focus_pred} 오분류 {mis}/{total} ({100*mis/total:.1f}%)")
    return cm


def resolve_source_checkpoint(animal: str, override: Optional[str]) -> Path:
    if override:
        p = Path(override)
    elif os.environ.get("CHECKPOINT", "").strip():
        p = Path(os.environ["CHECKPOINT"])
    else:
        p = Path(RandomSplitConfig.best_checkpoint(animal))
    if not p.is_file():
        raise FileNotFoundError(f"소스 체크포인트 없음: {p}")
    return p


def train_head_rebalance(args: argparse.Namespace) -> None:
    animal = args.animal.lower()
    device = args.device or get_device()
    batch_size = resolve_batch_size(args.batch_size)
    num_workers = resolve_num_workers(device, args.num_workers)
    use_amp = device == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    os.makedirs(RandomSplitConfig.OUTPUT_DIR, exist_ok=True)

    source_ckpt = resolve_source_checkpoint(animal, args.checkpoint)
    save_path = os.path.join(RandomSplitConfig.OUTPUT_DIR, f"{animal}_best_head_rebalance.pth")

    print("=" * 64)
    print(f"Head Rebalance — {animal.upper()}")
    print(f"  source: {source_ckpt}\n  save:   {save_path}")

    train_loader, val_loader, train_ds, val_ds, split_meta = create_random_split_dataloaders(
        animal_type=animal,
        batch_size=batch_size,
        img_size=args.img_size,
        num_workers=num_workers,
        use_sampler=False,
        mode="multitask",
        pin_memory=device == "cuda",
    )

    model = create_model(animal, pretrained=False, head_dropout=BaseTrainConfig.HEAD_DROPOUT)
    ckpt = torch.load(source_ckpt, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state)
    model.to(device)
    ema = create_ema(model, device)
    ema.module.load_state_dict(state)

    diseases = model.get_disease_names()
    head_w = parse_head_weights(args.head_weights, diseases)
    print("\n⚖️  head_loss_weight:", {d: head_w[d] for d in diseases})
    criterion_dict = build_rebalance_losses(train_ds, diseases, device).to(device)

    freeze_bb = args.freeze_backbone.lower() in ("1", "true", "yes")
    unfreeze_block_start: Optional[int] = None
    head_lr = args.lr
    if args.unfreeze_blocks > 0:
        set_backbone_frozen(model)
        unfreeze_block_start = unfreeze_backbone_last_blocks(model, args.unfreeze_blocks)
        freeze_bb = False
        if head_lr == 1e-3:
            head_lr = 1e-4
            print(f"  2-phase discriminative LR: head={head_lr}, backbone={args.backbone_lr}")

    optimizer = build_optimizer(
        model, head_lr,
        args.backbone_lr if args.unfreeze_blocks > 0 else None,
        args.unfreeze_blocks, freeze_bb and args.unfreeze_blocks == 0,
    )
    print(f"  trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best_val = float("inf")
    history: dict = {"epochs": [], "head_loss_weights": head_w}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = train_epoch_rebalance(
            model, train_loader, criterion_dict, optimizer, device, diseases, head_w,
            freeze_bb and args.unfreeze_blocks == 0,
            unfreeze_block_start if args.unfreeze_blocks > 0 else None,
            use_amp, scaler, ema,
        )
        infer = eval_model(model, ema)
        va = validate_epoch_rebalance(infer, val_loader, criterion_dict, device, diseases, head_w, use_amp)
        scheduler.step()
        print(f"Train loss={tr['loss']:.4f}  Val loss={va['loss']:.4f}  Val acc={_mean_disease_metric(va, '_acc'):.4f}")
        _print_disease_metrics("[Val]", va, diseases)
        cm = quick_top1_confusion_summary(infer, val_ds, val_loader, diseases, device)
        topk = evaluate_multitask_topk(infer, val_ds, val_loader, diseases, device)
        history["epochs"].append({"epoch": epoch, "val_loss": va["loss"], "topk": topk["topk_accuracy"]})
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_state_dict(ema, model),
                "val_loss": va["loss"],
                "val_acc_mean": _mean_disease_metric(va, "_acc"),
                "task": "multitask_head_rebalance",
                "animal_type": animal,
                "source_checkpoint": str(source_ckpt),
                "head_loss_weights": head_w,
                "split_meta": split_meta,
                "topk_accuracy": topk["topk_accuracy"],
            }, save_path)
            print(f"✓ Best saved: {save_path}")

    hist = os.path.join(RandomSplitConfig.OUTPUT_DIR, f"head_rebalance_{animal}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(hist, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    print(f"Done. history={hist}")


def main() -> None:
    p = argparse.ArgumentParser(description="Head rebalance training")
    p.add_argument("--animal", default=os.environ.get("ANIMAL_TYPE", "dog"), choices=["dog", "cat"])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="--unfreeze-blocks>0 일 때 백본 LR (헤드는 기본 1e-4)")
    p.add_argument("--freeze-backbone", default="true")
    p.add_argument("--head-weights", default=None)
    p.add_argument("--unfreeze-blocks", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--img-size", type=int, default=int(os.environ.get("IMG_SIZE", "300")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "32")))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    train_head_rebalance(p.parse_args())


if __name__ == "__main__":
    main()

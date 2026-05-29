"""
강아지 부위 그룹(5-class) 캐스케이드 1단계 학습.

기법: EMA, Mixup+CutMix(50:50), CosineAnnealingWarmRestarts

실행:
  ANIMAL_TYPE=dog python models/classifier/train_group.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os

import torch

from models.classifier.dataset_group import create_group_dataloader
from models.classifier.model_group import create_group_model, count_parameters
from models.classifier.losses import FocalLoss
from models.classifier.train_common import (
    _env_flag,
    create_ema,
    ema_state_dict,
    get_device,
    resolve_batch_size,
    resolve_num_workers,
    run_training_loop,
)


class GroupConfig:
    DOG_DATA_PATHS = ["eye_data/개/안구/일반", "eye_data/TL2/개/안구/일반"]
    CAT_DATA_PATHS = ["eye_data/TL2/고양이/안구/일반"]
    DOG_VAL_PATHS = ["eye_data/VL/개/안구/일반"]
    CAT_VAL_PATHS = ["eye_data/VL/고양이/안구/일반"]

    ANIMAL_TYPE = os.environ.get("ANIMAL_TYPE", "dog").strip().lower()
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
    USE_SAMPLER = True
    USE_MIXUP = _env_flag("USE_MIXUP", "1")
    USE_CUTMIX = _env_flag("USE_CUTMIX", "1")
    MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.2"))
    CUTMIX_ALPHA = float(os.environ.get("CUTMIX_ALPHA", "1.0"))
    MIX_CHOICE_PROB = float(os.environ.get("MIX_CHOICE_PROB", "0.5"))
    PATIENCE = 5
    NUM_WORKERS = 0
    OUTPUT_DIR = "models/classifier/checkpoints"
    RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", "auto").strip()
    SCHEDULER_TYPE = os.environ.get("SCHEDULER_TYPE", "warm_restarts").strip()


def train():
    cfg = GroupConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = get_device()
    use_amp = device == "cuda"
    from torch.amp import GradScaler
    scaler = GradScaler("cuda") if use_amp else None

    paths = (
        (cfg.DOG_DATA_PATHS, cfg.DOG_VAL_PATHS)
        if cfg.ANIMAL_TYPE == "dog"
        else (cfg.CAT_DATA_PATHS, cfg.CAT_VAL_PATHS)
    )
    nw = resolve_num_workers(device)
    pin = device == "cuda"
    bs = resolve_batch_size(cfg.BATCH_SIZE)

    print("=" * 60)
    print(f"부위 그룹 학습 ({cfg.ANIMAL_TYPE.upper()})")
    print(
        f"  EMA | Mixup α={cfg.MIXUP_ALPHA} + CutMix α={cfg.CUTMIX_ALPHA} "
        f"(50:50) | scheduler={cfg.SCHEDULER_TYPE}"
    )
    print("=" * 60)

    train_loader = create_group_dataloader(
        paths[0], cfg.ANIMAL_TYPE, bs, cfg.IMG_SIZE, True, nw, cfg.USE_SAMPLER, pin,
    )
    val_loader = create_group_dataloader(
        paths[1], cfg.ANIMAL_TYPE, bs, cfg.IMG_SIZE, False, nw, False, pin,
    )

    model = create_group_model(cfg.ANIMAL_TYPE, head_dropout=cfg.HEAD_DROPOUT).to(device)
    ema = create_ema(model, device)
    class_names = model.class_names

    best_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_group_best.pth")
    resume = cfg.RESUME_CHECKPOINT
    skip_p1 = False
    if resume and resume.lower() == "auto" and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.module.load_state_dict(ckpt["model_state_dict"])
        skip_p1 = True
        print(f"Resume: {best_path}")
    elif resume and resume not in ("0", "false") and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.module.load_state_dict(ckpt["model_state_dict"])
        skip_p1 = True

    cw = train_loader.dataset.get_class_weights().to(device)
    criterion = FocalLoss(
        gamma=cfg.FOCAL_GAMMA,
        alpha=cw / cw.mean().clamp(min=1e-6),
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    print(f"Params: {count_parameters(model):,}")

    cm_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_group_confusion_matrix.png")
    hist_prefix = os.path.join(cfg.OUTPUT_DIR, f"training_history_{cfg.ANIMAL_TYPE}_group")

    run_training_loop(
        model, train_loader, val_loader, criterion, cfg, device,
        class_names, best_path, cm_path, hist_prefix, use_amp, scaler, skip_p1,
        ema=ema,
        scheduler_type=cfg.SCHEDULER_TYPE,
        use_cutmix=cfg.USE_CUTMIX,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        mix_choice_prob=cfg.MIX_CHOICE_PROB,
    )

    final = os.path.join(cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_group_final.pth")
    torch.save({
        "model_state_dict": ema_state_dict(ema, model),
        "task": "group_classifier",
        "ema_decay": 0.9998,
    }, final)
    print(f"Final: {final}")


if __name__ == "__main__":
    train()

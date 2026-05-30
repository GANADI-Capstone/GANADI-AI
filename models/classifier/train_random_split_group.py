"""
TL random split 부위 그룹(5-class) 학습.

VL 미사용 · crop_D* 그룹 80/20 · MAX_PER_CLASS cap · device subset 평가.

실행 (프로젝트 루트):
  ANIMAL_TYPE=dog MAX_PER_CLASS=5000 python models/classifier/train_random_split_group.py
  ANIMAL_TYPE=cat MAX_PER_CLASS=5000 python models/classifier/train_random_split_group.py

환경변수:
  SPLIT_SEED=42
  VAL_RATIO=0.2
  MAX_PER_CLASS=5000
  USE_GROUP_SPLIT=1
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import os
from datetime import datetime

import torch

from models.classifier.dataset_random_split import create_random_split_dataloaders
from models.classifier.losses import FocalLoss
from models.classifier.model_group import create_group_model, count_parameters
from models.classifier.train_common import (
    create_ema,
    ema_state_dict,
    eval_model,
    get_device,
    resolve_batch_size,
    resolve_num_workers,
    run_training_loop,
)
from models.classifier.train_group import GroupConfig
from models.classifier.train_random_split import evaluate_device_subsets_single


class RandomSplitGroupConfig(GroupConfig):
    SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
    VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))
    MAX_PER_CLASS = os.environ.get("MAX_PER_CLASS", "").strip()

    @staticmethod
    def best_checkpoint(animal: str) -> str:
        return os.path.join(
            RandomSplitGroupConfig.OUTPUT_DIR,
            f"{animal}_group_best_random_split.pth",
        )

    @staticmethod
    def final_checkpoint(animal: str) -> str:
        return os.path.join(
            RandomSplitGroupConfig.OUTPUT_DIR,
            f"{animal}_group_final_random_split.pth",
        )


def train() -> None:
    cfg = RandomSplitGroupConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = get_device()
    use_amp = device == "cuda"
    from torch.amp import GradScaler

    scaler = GradScaler("cuda") if use_amp else None
    nw = resolve_num_workers(device)
    pin = device == "cuda"
    bs = resolve_batch_size(cfg.BATCH_SIZE)

    print("=" * 64)
    print(f"부위 그룹 — TL Random Split ({cfg.ANIMAL_TYPE.upper()})")
    print("=" * 64)
    print(f"  VL 미사용 | SPLIT_SEED={cfg.SPLIT_SEED} | VAL_RATIO={cfg.VAL_RATIO}")
    if cfg.MAX_PER_CLASS:
        print(f"  MAX_PER_CLASS={cfg.MAX_PER_CLASS}")
    print(
        f"  EMA | Mixup α={cfg.MIXUP_ALPHA} + CutMix α={cfg.CUTMIX_ALPHA} "
        f"(50:50) | scheduler={cfg.SCHEDULER_TYPE}"
    )

    train_loader, val_loader, train_ds, val_ds, split_meta = create_random_split_dataloaders(
        cfg.ANIMAL_TYPE,
        batch_size=bs,
        img_size=cfg.IMG_SIZE,
        num_workers=nw,
        use_sampler=cfg.USE_SAMPLER,
        pin_memory=pin,
        mode="group",
    )
    class_names = train_ds.class_names

    model = create_group_model(cfg.ANIMAL_TYPE, head_dropout=cfg.HEAD_DROPOUT).to(device)
    ema = create_ema(model, device)

    best_path = cfg.best_checkpoint(cfg.ANIMAL_TYPE)
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

    cw = train_ds.get_class_weights().to(device)
    criterion = FocalLoss(
        gamma=cfg.FOCAL_GAMMA,
        alpha=cw / cw.mean().clamp(min=1e-6),
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    print(f"Params: {count_parameters(model):,}")

    cm_path = os.path.join(
        cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_group_confusion_matrix_random_split.png",
    )
    hist_prefix = os.path.join(
        cfg.OUTPUT_DIR, f"training_history_{cfg.ANIMAL_TYPE}_group_random_split",
    )

    run_training_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        cfg,
        device,
        class_names,
        best_path,
        cm_path,
        hist_prefix,
        use_amp,
        scaler,
        skip_p1,
        ema=ema,
        scheduler_type=cfg.SCHEDULER_TYPE,
        use_cutmix=cfg.USE_CUTMIX,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        mix_choice_prob=cfg.MIX_CHOICE_PROB,
        task="group_random_split",
        extra_ckpt={
            "animal_type": cfg.ANIMAL_TYPE,
            "split_meta": split_meta,
            "max_per_class": split_meta.get("max_per_class"),
        },
    )

    infer = eval_model(model, ema)
    device_report = evaluate_device_subsets_single(infer, val_ds, val_loader, device)

    final_path = cfg.final_checkpoint(cfg.ANIMAL_TYPE)
    torch.save(
        {
            "model_state_dict": ema_state_dict(ema, model),
            "task": "group_random_split",
            "animal_type": cfg.ANIMAL_TYPE,
            "class_names": class_names,
            "split_meta": split_meta,
            "device_report": device_report,
            "ema_decay": 0.9998,
        },
        final_path,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        cfg.OUTPUT_DIR,
        f"device_report_{cfg.ANIMAL_TYPE}_group_random_split_{ts}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"split_meta": split_meta, "device_report": device_report},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 64)
    print("✅ Group TL Random Split 학습 완료")
    print("=" * 64)
    print(f"  Best:  {best_path}")
    print(f"  Final: {final_path}")
    print(f"  Device report: {report_path}")


if __name__ == "__main__":
    train()

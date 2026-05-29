"""
강아지 부위별 세부 질환 캐스케이드 2단계 학습.

학습 대상: 안검, 기타 (각막 세부 분류는 제외 — 추론 시 '각막계 질환 의심'만 출력)

전문가별 프로필:
  안검 — EMA, CBAM, RandAugment
  기타 — EMA만 (기본 설정)

Colab (A100 + Drive 100GB):
  ANIMAL_TYPE=dog python models/classifier/train_binary.py
  ANIMAL_TYPE=dog python models/classifier/train_group.py
  ANIMAL_TYPE=dog GROUP_NAME=안검 python models/classifier/train_subgroup.py
  ANIMAL_TYPE=dog GROUP_NAME=기타 python models/classifier/train_subgroup.py

실행:
  ANIMAL_TYPE=dog GROUP_NAME=안검 python models/classifier/train_subgroup.py
  ANIMAL_TYPE=dog GROUP_NAME=기타 python models/classifier/train_subgroup.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
from dataclasses import dataclass
from typing import Optional

import torch

from models.classifier.dataset_group import GROUP_SLUG
from models.classifier.dataset_subgroup import (
    create_subgroup_dataloader,
    get_subgroup_diseases,
    subgroup_trainable,
)
from models.classifier.model_subgroup import create_subgroup_model, count_parameters
from models.classifier.model_subgroup_cbam import create_subgroup_cbam_model
from models.classifier.losses import FocalLoss
from models.classifier.losses_supcon import SupConLoss
from models.classifier.train_common import (
    _env_flag,
    create_ema,
    ema_state_dict,
    get_device,
    resolve_batch_size,
    resolve_img_size,
    resolve_num_workers,
    run_training_loop,
)
from models.classifier.train_group import GroupConfig


@dataclass
class SubgroupProfile:
    use_cbam: bool = False
    use_projection: bool = False
    use_supcon: bool = False
    use_rand_augment: bool = False
    use_cutmix: bool = False
    mixup_alpha: float = 0.1
    cutmix_alpha: float = 1.0
    mix_choice_prob: float = 0.5
    head_dropout: float = 0.3
    label_smoothing: float = 0.1
    img_size: int = 300
    supcon_weight: float = 0.5
    ce_weight: float = 0.5
    supcon_temperature: float = 0.07
    scheduler_type: str = "cosine"


# 세부 모델 학습 제외 (추론: predict_cascade.py 에서 부위 수준 의심만 출력)
SUBGROUP_SKIP_TRAINING = frozenset({"각막"})

SUBGROUP_PROFILES = {
    "안검": SubgroupProfile(
        use_cbam=True,
        use_rand_augment=True,
        mixup_alpha=0.1,
        head_dropout=0.3,
        label_smoothing=0.1,
    ),
    "기타": SubgroupProfile(
        mixup_alpha=0.1,
        head_dropout=0.3,
        label_smoothing=0.1,
    ),
}


class SubgroupConfig(GroupConfig):
    GROUP_NAME = os.environ.get("GROUP_NAME", "안검").strip()

    def apply_profile(self, profile: SubgroupProfile) -> None:
        self.HEAD_DROPOUT = profile.head_dropout
        self.LABEL_SMOOTHING = profile.label_smoothing
        self.MIXUP_ALPHA = profile.mixup_alpha
        self.USE_MIXUP = _env_flag("USE_MIXUP", "1")
        self.USE_CUTMIX = profile.use_cutmix
        self.CUTMIX_ALPHA = profile.cutmix_alpha
        self.MIX_CHOICE_PROB = profile.mix_choice_prob
        self.IMG_SIZE = resolve_img_size(profile.img_size)
        self.SCHEDULER_TYPE = profile.scheduler_type
        self._profile = profile


def subgroup_checkpoint_slug(group_name: str) -> str:
    return GROUP_SLUG.get(group_name, group_name)


def get_profile(group_name: str) -> SubgroupProfile:
    if group_name not in SUBGROUP_PROFILES:
        print(f"⚠ '{group_name}' 전용 프로필 없음 — 기타(EMA만) 설정 사용")
        return SUBGROUP_PROFILES["기타"]
    return SUBGROUP_PROFILES[group_name]


def create_model_for_profile(profile: SubgroupProfile, ds, group_name: str):
    if profile.use_cbam:
        return create_subgroup_cbam_model(
            num_classes=ds.num_classes,
            class_names=ds.class_names,
            group_name=group_name,
            head_dropout=profile.head_dropout,
            use_projection=profile.use_projection,
        )
    return create_subgroup_model(
        num_classes=ds.num_classes,
        class_names=ds.class_names,
        group_name=group_name,
        head_dropout=profile.head_dropout,
    )


def train():
    cfg = SubgroupConfig()
    profile = get_profile(cfg.GROUP_NAME)
    cfg.apply_profile(profile)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if cfg.GROUP_NAME in SUBGROUP_SKIP_TRAINING:
        print(
            f"⚠ GROUP_NAME='{cfg.GROUP_NAME}' — 세부 subgroup 학습 생략.\n"
            "  추론: predict_cascade.py → '각막계 질환이 의심됩니다. 수의사 상담 권장.'"
        )
        return

    if not subgroup_trainable(cfg.GROUP_NAME, cfg.ANIMAL_TYPE):
        diseases = get_subgroup_diseases(cfg.ANIMAL_TYPE, cfg.GROUP_NAME)
        print(
            f"⚠ GROUP_NAME='{cfg.GROUP_NAME}' — 질환 {diseases} (1개)\n"
            "  세부 분류 모델 불필요. 그룹 분류 결과로 질환 확정."
        )
        return

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
    slug = subgroup_checkpoint_slug(cfg.GROUP_NAME)

    aug_preset = "rand" if profile.use_rand_augment else "train"
    print("=" * 60)
    print(f"Subgroup 학습 [{cfg.GROUP_NAME}] ({cfg.ANIMAL_TYPE.upper()})")
    print(f"  IMG_SIZE={cfg.IMG_SIZE} | Dropout={cfg.HEAD_DROPOUT} | LS={cfg.LABEL_SMOOTHING}")
    feats = ["EMA"]
    if profile.use_cbam:
        feats.append("CBAM")
    if profile.use_rand_augment:
        feats.append("RandAugment")
    if profile.use_supcon:
        feats.append(f"SupCon(τ={profile.supcon_temperature})")
    if profile.use_cutmix:
        feats.append(f"Mixup α={profile.mixup_alpha}+CutMix α={profile.cutmix_alpha}")
    else:
        feats.append(f"Mixup α={profile.mixup_alpha}")
    print(f"  기법: {', '.join(feats)}")
    print("=" * 60)

    train_loader = create_subgroup_dataloader(
        paths[0], cfg.ANIMAL_TYPE, cfg.GROUP_NAME,
        bs, cfg.IMG_SIZE, True, nw, cfg.USE_SAMPLER, pin,
        aug_preset=aug_preset,
    )
    val_loader = create_subgroup_dataloader(
        paths[1], cfg.ANIMAL_TYPE, cfg.GROUP_NAME,
        bs, cfg.IMG_SIZE, False, nw, False, pin,
        aug_preset="train",
    )

    ds = train_loader.dataset
    model = create_model_for_profile(profile, ds, cfg.GROUP_NAME).to(device)
    ema = create_ema(model, device)

    best_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_{slug}_best.pth")
    skip_p1 = False
    resume = cfg.RESUME_CHECKPOINT
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

    cw = ds.get_class_weights().to(device)
    criterion = FocalLoss(
        gamma=cfg.FOCAL_GAMMA,
        alpha=cw / cw.mean().clamp(min=1e-6),
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    supcon_criterion: Optional[SupConLoss] = None
    if profile.use_supcon:
        supcon_criterion = SupConLoss(temperature=profile.supcon_temperature)
        print(
            f"✓ SupCon Loss (CE:{profile.ce_weight} + SupCon:{profile.supcon_weight}, "
            f"proj 1536→128, τ={profile.supcon_temperature})"
        )

    print(f"Params: {count_parameters(model):,}")

    cm_path = os.path.join(
        cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_{slug}_confusion_matrix.png",
    )
    hist_prefix = os.path.join(
        cfg.OUTPUT_DIR, f"training_history_{cfg.ANIMAL_TYPE}_{slug}",
    )

    run_training_loop(
        model, train_loader, val_loader, criterion, cfg, device,
        ds.class_names, best_path, cm_path, hist_prefix,
        use_amp, scaler, skip_p1,
        task="subgroup_classifier",
        extra_ckpt={
            "group_name": cfg.GROUP_NAME,
            "animal_type": cfg.ANIMAL_TYPE,
            "img_size": cfg.IMG_SIZE,
            "profile": cfg.GROUP_NAME,
        },
        ema=ema,
        supcon_criterion=supcon_criterion,
        supcon_weight=profile.supcon_weight,
        ce_weight=profile.ce_weight,
        scheduler_type=cfg.SCHEDULER_TYPE,
        use_cutmix=cfg.USE_CUTMIX,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        mix_choice_prob=cfg.MIX_CHOICE_PROB,
    )

    final = os.path.join(cfg.OUTPUT_DIR, f"{cfg.ANIMAL_TYPE}_{slug}_final.pth")
    torch.save({
        "model_state_dict": ema_state_dict(ema, model),
        "task": "subgroup_classifier",
        "group_name": cfg.GROUP_NAME,
        "class_names": ds.class_names,
        "img_size": cfg.IMG_SIZE,
        "ema_decay": 0.9998,
    }, final)
    print(f"Final: {final}")


if __name__ == "__main__":
    train()

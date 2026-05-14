"""
EfficientNet-B3 멀티태스크 질환 분류 모델 학습 스크립트

실행 (프로젝트 루트에서):
  source venv/bin/activate
  python3 models/classifier/train.py

맥북 발열 완화 (GPU 끄고 CPU 학습, 매우 느림):
  FORCE_CPU_TRAINING=1 python3 models/classifier/train.py

※ 스크립트 경로로 실행 시에도 동작하도록 저장소 루트를 sys.path 에 넣습니다.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict
import json

from models.classifier.model import create_model, count_parameters
from models.classifier.dataset import create_dataloader
from models.classifier.losses import build_per_disease_losses


# 설정
class Config:
    # 데이터 - Training
    DOG_DATA_PATHS = [
        "eye_data/개/안구/일반",
        "eye_data/TL2/개/안구/일반"
    ]
    CAT_DATA_PATHS = [
        "eye_data/TL2/고양이/안구/일반"
    ]
    
    # 데이터 - Validation
    DOG_VAL_PATHS = [
        "eye_data/VL/개/안구/일반"
    ]
    CAT_VAL_PATHS = [
        "eye_data/VL/고양이/안구/일반"
    ]
    
    # 학습
    ANIMAL_TYPE = "cat"  # "dog" → "cat"
    EPOCHS = 20
    BATCH_SIZE = 32
    IMG_SIZE = 224
    LR = 1e-5  # 1e-4 → 1e-5 (낮춤)
    WEIGHT_DECAY = 1e-4
    
    # 클래스 불균형 · 손실 · 증강 (고양이 결막염 등 개선 실험)
    # LOSS_TYPE: ce | weighted_ce | focal
    LOSS_TYPE = "focal"
    USE_CLASS_WEIGHTS = True
    FOCAL_GAMMA = 2.0
    # aug_preset: default | cat_phone (색·압축·블러 강화 — 강아지 학습 시 자동으로 default)
    AUG_PRESET = "cat_phone"
    USE_SAMPLER = False
    # 고양이만 True 권장: WeightedRandomSampler + 결막염 소수 클래스 부스트
    USE_SAMPLER_FOR_CAT = True
    SAMPLER_BOOST_DISEASE = "결막염"
    SAMPLER_BOOST_FACTOR = 2.5
    
    # Early Stopping
    PATIENCE = 5
    
    # 디바이스
    DEVICE = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    # 저장
    OUTPUT_DIR = "models/classifier/checkpoints"
    
    # Wandb (선택사항)
    USE_WANDB = False
    WANDB_PROJECT = "eye-disease-classification"


def get_device() -> str:
    """디바이스 선택.

    발열 완화: FORCE_CPU_TRAINING=1 이면 MPS/CUDA 를 쓰지 않고 CPU 만 사용 (느리지만 GPU 발열 없음).
    """
    if os.environ.get("FORCE_CPU_TRAINING", "").strip() in ("1", "true", "yes"):
        print("⚠ CPU 학습 모드 (FORCE_CPU_TRAINING=1) — GPU 미사용, 발열·속도 트레이드오프")
        return "cpu"
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) 사용")
        return "mps"
    if torch.cuda.is_available():
        print("✓ CUDA (NVIDIA GPU) 사용")
        return "cuda"
    print("⚠ CPU 사용 (학습이 느릴 수 있습니다)")
    return "cpu"


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_dict: nn.ModuleDict,
    optimizer: optim.Optimizer,
    device: str,
    diseases: list,
) -> Dict[str, float]:
    """1 epoch 학습"""
    model.train()

    total_loss = 0.0
    disease_losses = {d: 0.0 for d in diseases}
    disease_corrects = {d: 0 for d in diseases}
    disease_totals = {d: 0 for d in diseases}

    progress = tqdm(dataloader, desc="Training")

    for images, labels in progress:
        images = images.to(device)

        outputs = model(images)

        loss = torch.tensor(0.0, device=device)
        batch_losses = {}

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

            disease_losses[disease] += disease_loss.item() * valid_mask.sum().item()
            _, preds = torch.max(valid_outputs, 1)
            disease_corrects[disease] += (preds == valid_labels).sum().item()
            disease_totals[disease] += valid_mask.sum().item()

            batch_losses[disease] = disease_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)

    metrics = {"loss": avg_loss}

    for disease in diseases:
        if disease_totals[disease] > 0:
            disease_loss = disease_losses[disease] / disease_totals[disease]
            disease_acc = disease_corrects[disease] / disease_totals[disease]
            metrics[f"{disease}_loss"] = disease_loss
            metrics[f"{disease}_acc"] = disease_acc

    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_dict: nn.ModuleDict,
    device: str,
    diseases: list,
) -> Dict[str, float]:
    """검증"""
    model.eval()

    total_loss = 0.0
    disease_losses = {d: 0.0 for d in diseases}
    disease_corrects = {d: 0 for d in diseases}
    disease_totals = {d: 0 for d in diseases}

    progress = tqdm(dataloader, desc="Validation")

    for images, labels in progress:
        images = images.to(device)

        outputs = model(images)

        loss = torch.tensor(0.0, device=device)

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

            disease_losses[disease] += disease_loss.item() * valid_mask.sum().item()
            _, preds = torch.max(valid_outputs, 1)
            disease_corrects[disease] += (preds == valid_labels).sum().item()
            disease_totals[disease] += valid_mask.sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    metrics = {"loss": avg_loss}

    for disease in diseases:
        if disease_totals[disease] > 0:
            disease_loss = disease_losses[disease] / disease_totals[disease]
            disease_acc = disease_corrects[disease] / disease_totals[disease]
            metrics[f"{disease}_loss"] = disease_loss
            metrics[f"{disease}_acc"] = disease_acc

    return metrics


def train():
    """메인 학습 함수"""
    config = Config()
    
    print("=" * 60)
    print("EfficientNet-B3 멀티태스크 질환 분류 모델 학습")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 디바이스
    device = get_device()
    
    # 데이터 경로 설정
    if config.ANIMAL_TYPE.lower() == "dog":
        train_paths = config.DOG_DATA_PATHS
        val_paths = config.DOG_VAL_PATHS
    else:
        train_paths = config.CAT_DATA_PATHS
        val_paths = config.CAT_VAL_PATHS
    
    print(f"\n⚙️  설정:")
    print(f"  - 동물: {config.ANIMAL_TYPE.upper()}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Image Size: {config.IMG_SIZE}")
    print(f"  - Learning Rate: {config.LR}")
    print(f"  - Device: {device}")
    print(f"  - Loss: {config.LOSS_TYPE} (class_weights={config.USE_CLASS_WEIGHTS}, γ={config.FOCAL_GAMMA})")
    print(f"  - Aug preset: {config.AUG_PRESET}")

    aug_preset = config.AUG_PRESET
    if config.ANIMAL_TYPE.lower() == "dog" and aug_preset == "cat_phone":
        aug_preset = "default"
        print("⚠️  강아지 학습: AUG_PRESET 을 'default' 로 사용합니다.")

    use_sampler = bool(config.USE_SAMPLER)
    if config.ANIMAL_TYPE.lower() == "cat" and getattr(
        config, "USE_SAMPLER_FOR_CAT", False
    ):
        use_sampler = True
    boost_dis = config.SAMPLER_BOOST_DISEASE if use_sampler else None
    print(
        f"  - WeightedSampler: {'ON' if use_sampler else 'OFF'}"
        + (f" (boost={boost_dis}×{config.SAMPLER_BOOST_FACTOR})" if use_sampler else "")
    )

    print(f"\n📂 데이터 경로:")
    print(f"  - Train: {train_paths}")
    print(f"  - Val:   {val_paths}")

    pin_mem = device == "cuda"

    print(f"\n📊 데이터 로딩...")
    train_loader = create_dataloader(
        data_paths=train_paths,
        animal_type=config.ANIMAL_TYPE,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        is_training=True,
        num_workers=0,
        use_sampler=use_sampler,
        aug_preset=aug_preset,
        sampler_boost_disease=boost_dis,
        sampler_boost_factor=config.SAMPLER_BOOST_FACTOR,
        pin_memory=pin_mem,
    )

    val_loader = create_dataloader(
        data_paths=val_paths,
        animal_type=config.ANIMAL_TYPE,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        is_training=False,
        num_workers=0,
        use_sampler=False,
        aug_preset="default",
        pin_memory=pin_mem,
    )

    print(f"\n🔧 모델 생성...")
    model = create_model(animal_type=config.ANIMAL_TYPE, pretrained=True)
    model = model.to(device)

    print(f"학습 가능 파라미터: {count_parameters(model):,}")

    diseases = model.get_disease_names()

    criterion_dict = build_per_disease_losses(
        train_loader.dataset,
        diseases,
        config.LOSS_TYPE,
        device=device,
        use_class_weights=config.USE_CLASS_WEIGHTS,
        focal_gamma=config.FOCAL_GAMMA,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Wandb 초기화 (선택사항)
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            config={
                "animal_type": config.ANIMAL_TYPE,
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "lr": config.LR,
            }
        )
    
    # 학습 루프
    print(f"\n🚀 학습 시작...\n")

    best_val_loss = float("inf")
    patience_counter = 0
    save_path = ""
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion_dict, optimizer, device, diseases
        )

        val_metrics = validate_epoch(
            model, val_loader, criterion_dict, device, diseases
        )
        
        # Scheduler step
        scheduler.step()
        
        # 출력
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f}")
        print(f"[Val]   Loss: {val_metrics['loss']:.4f}")
        
        # Wandb 로깅
        if config.USE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                "val_loss": val_metrics['loss'],
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Best 모델 저장
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            save_path = os.path.join(
                config.OUTPUT_DIR,
                f"{config.ANIMAL_TYPE}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config.__dict__
            }, save_path)
            
            print(f"✓ Best 모델 저장: {save_path}")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️  Early Stopping (patience={config.PATIENCE})")
            break
    
    # 최종 모델 저장
    final_path = os.path.join(
        config.OUTPUT_DIR,
        f"{config.ANIMAL_TYPE}_final.pth"
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__
    }, final_path)
    
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"\n📂 저장 위치:")
    print(f"  - Best: {save_path or '(저장 없음)'}")
    print(f"  - Final: {final_path}")
    
    if config.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()

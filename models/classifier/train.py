"""
EfficientNet-B3 멀티태스크 질환 분류 모델 학습 스크립트
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict
import json

from models.classifier.model import create_model, count_parameters
from models.classifier.dataset import create_dataloader


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
    
    # 클래스 불균형 처리
    USE_SAMPLER = False  # True → False (오버피팅 방지)
    
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


def get_device():
    """디바이스 확인 및 출력"""
    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) 사용")
        return "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA (NVIDIA GPU) 사용")
        return "cuda"
    else:
        print("⚠ CPU 사용 (학습이 느릴 수 있습니다)")
        return "cpu"


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    diseases: list
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
        
        # Forward
        outputs = model(images)
        
        # 질환별 Loss 계산
        loss = 0
        batch_losses = {}
        
        for disease in diseases:
            disease_labels = labels[disease].to(device)
            disease_outputs = outputs[disease]
            
            # 유효한 샘플만 (label >= 0)
            valid_mask = disease_labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            valid_labels = disease_labels[valid_mask]
            valid_outputs = disease_outputs[valid_mask]
            
            # CrossEntropy Loss
            disease_loss = criterion(valid_outputs, valid_labels)
            loss += disease_loss
            
            # 통계
            disease_losses[disease] += disease_loss.item() * valid_mask.sum().item()
            _, preds = torch.max(valid_outputs, 1)
            disease_corrects[disease] += (preds == valid_labels).sum().item()
            disease_totals[disease] += valid_mask.sum().item()
            
            batch_losses[disease] = disease_loss.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress bar 업데이트
        progress.set_postfix({
            'loss': loss.item()
        })
    
    # Epoch 통계 계산
    avg_loss = total_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss
    }
    
    for disease in diseases:
        if disease_totals[disease] > 0:
            disease_loss = disease_losses[disease] / disease_totals[disease]
            disease_acc = disease_corrects[disease] / disease_totals[disease]
            metrics[f'{disease}_loss'] = disease_loss
            metrics[f'{disease}_acc'] = disease_acc
    
    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    diseases: list
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
        
        # Forward
        outputs = model(images)
        
        # 질환별 Loss 계산
        loss = 0
        
        for disease in diseases:
            disease_labels = labels[disease].to(device)
            disease_outputs = outputs[disease]
            
            # 유효한 샘플만
            valid_mask = disease_labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            valid_labels = disease_labels[valid_mask]
            valid_outputs = disease_outputs[valid_mask]
            
            # Loss
            disease_loss = criterion(valid_outputs, valid_labels)
            loss += disease_loss
            
            # 통계
            disease_losses[disease] += disease_loss.item() * valid_mask.sum().item()
            _, preds = torch.max(valid_outputs, 1)
            disease_corrects[disease] += (preds == valid_labels).sum().item()
            disease_totals[disease] += valid_mask.sum().item()
        
        total_loss += loss.item()
    
    # 통계 계산
    avg_loss = total_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss
    }
    
    for disease in diseases:
        if disease_totals[disease] > 0:
            disease_loss = disease_losses[disease] / disease_totals[disease]
            disease_acc = disease_corrects[disease] / disease_totals[disease]
            metrics[f'{disease}_loss'] = disease_loss
            metrics[f'{disease}_acc'] = disease_acc
    
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
    print(f"  - WeightedSampler: {'ON' if config.USE_SAMPLER else 'OFF'}")
    
    print(f"\n📂 데이터 경로:")
    print(f"  - Train: {train_paths}")
    print(f"  - Val:   {val_paths}")
    
    # DataLoader 생성
    print(f"\n📊 데이터 로딩...")
    train_loader = create_dataloader(
        data_paths=train_paths,
        animal_type=config.ANIMAL_TYPE,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        is_training=True,
        num_workers=0,  # 4 → 0 (MPS에서 더 빠름)
        use_sampler=config.USE_SAMPLER  # WeightedRandomSampler 사용
    )
    
    val_loader = create_dataloader(
        data_paths=val_paths,
        animal_type=config.ANIMAL_TYPE,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        is_training=False,
        num_workers=0,  # 4 → 0 (MPS에서 더 빠름)
        use_sampler=False  # Validation은 sampler 미사용
    )
    
    # 모델 생성
    print(f"\n🔧 모델 생성...")
    model = create_model(animal_type=config.ANIMAL_TYPE, pretrained=True)
    model = model.to(device)
    
    print(f"학습 가능 파라미터: {count_parameters(model):,}")
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    diseases = model.get_disease_names()
    
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
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, diseases
        )
        
        # Validation
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, diseases
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
    print(f"  - Best: {save_path}")
    print(f"  - Final: {final_path}")
    
    if config.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    train()

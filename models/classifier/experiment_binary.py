"""
중증도 통합 실험 스크립트
백내장, 궤양성, 비궤양성을 유/무 이진 분류로 변경
"""

import torch
from model import create_model
from dataset import EyeDiseaseDataset, create_dataloader
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === 설정 ===
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

# 중증도 통합할 질환
SIMPLIFY_DISEASES = ['백내장', '궤양성각막질환', '비궤양성각막질환']

print("\n" + "="*70)
print("  중증도 통합 실험 (Binary Classification)")
print("="*70 + "\n")

# === 데이터셋 수정 ===
class SimplifiedDataset(EyeDiseaseDataset):
    """중증도를 유/무로 단순화한 데이터셋"""
    
    def __getitem__(self, idx):
        image, labels = super().__getitem__(idx)
        
        # 중증도 통합
        for disease in SIMPLIFY_DISEASES:
            if disease in labels:
                label_value = labels[disease].item()
                # 0(무)은 그대로, 1 이상(중증도)은 모두 1(유)로
                if label_value > 0:
                    labels[disease] = torch.tensor(1)
        
        return image, labels

# === 모델 생성 (헤드 크기 조정) ===
def create_simplified_model(species='dog'):
    """중증도 통합 모델 생성 - 기존 모델 활용"""
    # 기존 모델 그대로 사용 (질환별 헤드는 이미 존재)
    # 단, 데이터셋에서 라벨을 0/1로 변환하므로
    # 중증도 클래스(2, 3 등)는 사용되지 않음
    model = create_model(species, pretrained=False)
    return model

# === 학습 ===
def train_simplified():
    print("1️⃣ 데이터 로딩...")
    
    # Transform 정의
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_dataset = SimplifiedDataset(
        data_paths=['../../eye_data/TL2/개/안구/일반'],
        animal_type='dog',
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = SimplifiedDataset(
        data_paths=['../../eye_data/VL/개/안구/일반'],
        animal_type='dog',
        transform=val_transform,
        is_training=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    print(f"   학습: {len(train_dataset)}장")
    print(f"   검증: {len(val_dataset)}장\n")
    
    # 모델
    print("2️⃣ 모델 생성...")
    model = create_simplified_model('dog')
    model = model.to(DEVICE)
    
    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    print("\n3️⃣ 학습 시작...\n")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = 0
            for disease in model.get_disease_names():
                dlabels = labels[disease].to(DEVICE)
                valid = dlabels >= 0
                if valid.sum() == 0:
                    continue
                loss += criterion(outputs[disease][valid], dlabels[valid])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                
                for disease in model.get_disease_names():
                    dlabels = labels[disease].to(DEVICE)
                    valid = dlabels >= 0
                    if valid.sum() == 0:
                        continue
                    preds = torch.argmax(outputs[disease][valid], dim=1)
                    val_correct += (preds == dlabels[valid]).sum().item()
                    val_total += valid.sum().item()
        
        val_acc = val_correct / val_total * 100
        
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Loss: {train_loss/len(train_loader):.4f} - "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, 'checkpoints/dog_simplified_best.pth')
            print(f"   ✓ Best 모델 저장!")
    
    print(f"\n✅ 학습 완료! Best Val Acc: {best_val_acc:.2f}%")
    
    # 질환별 상세 평가
    print("\n4️⃣ 질환별 성능 측정...\n")
    model.load_state_dict(
        torch.load('checkpoints/dog_simplified_best.pth')['model_state_dict']
    )
    model.eval()
    
    disease_correct = {d: 0 for d in model.get_disease_names()}
    disease_total = {d: 0 for d in model.get_disease_names()}
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            for disease in model.get_disease_names():
                dlabels = labels[disease].to(DEVICE)
                valid = dlabels >= 0
                if valid.sum() == 0:
                    continue
                preds = torch.argmax(outputs[disease][valid], dim=1)
                disease_correct[disease] += (preds == dlabels[valid]).sum().item()
                disease_total[disease] += valid.sum().item()
    
    print(f"{'질환명':<20} {'정확도':<12} {'개선 전':<12} {'개선'}")
    print("-" * 60)
    
    # 기존 성능 (참고용)
    baseline = {
        '백내장': 76.0,
        '궤양성각막질환': 79.3,
        '비궤양성각막질환': 79.3,
    }
    
    for disease in model.get_disease_names():
        acc = disease_correct[disease] / disease_total[disease] * 100
        old_acc = baseline.get(disease, 0)
        improvement = acc - old_acc if old_acc > 0 else 0
        
        print(f"{disease:<20} {acc:>6.1f}%      "
              f"{old_acc:>6.1f}%      {improvement:>+5.1f}%")

if __name__ == '__main__':
    train_simplified()

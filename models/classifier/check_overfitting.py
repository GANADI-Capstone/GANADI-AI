"""
과적합(Overfitting) 검증 스크립트
- 학습 데이터 vs 검증 데이터 성능 비교
- 데이터 누수 체크
- Loss 추이 분석
"""

import torch
from model import create_model
from dataset import create_dataloader
from collections import defaultdict
import matplotlib.pyplot as plt

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("\n" + "="*70)
print("  과적합(Overfitting) 검증")
print("="*70 + "\n")

# 1. 모델 로드
print("1️⃣ 모델 로딩...")
model = create_model('dog', pretrained=False)
ckpt = torch.load('checkpoints/dog_best.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

print(f"   ✓ 체크포인트 Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"   ✓ Best Val Loss: {ckpt.get('best_val_loss', 'N/A'):.4f}" if 'best_val_loss' in ckpt else "")

# 2. 학습 데이터 성능
print("\n2️⃣ 학습 데이터 성능 측정...")
train_loader = create_dataloader(
    data_paths=['../../eye_data/TL1/개/안구/일반', '../../eye_data/TL2/개/안구/일반'],
    animal_type='dog',
    batch_size=32,
    img_size=224,
    is_training=False,  # augmentation 끄기
    num_workers=0,
    use_sampler=False
)

diseases = model.get_disease_names()
train_correct = defaultdict(int)
train_total = defaultdict(int)

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)
        for disease in diseases:
            dlabels = labels[disease].to(device)
            valid = dlabels >= 0
            if valid.sum() == 0:
                continue
            preds = torch.argmax(outputs[disease][valid], dim=1)
            gt = dlabels[valid]
            train_correct[disease] += (preds == gt).sum().item()
            train_total[disease] += valid.sum().item()

# 3. 검증 데이터 성능
print("\n3️⃣ 검증 데이터 성능 측정...")
val_loader = create_dataloader(
    data_paths=['../../eye_data/VL/개/안구/일반'],
    animal_type='dog',
    batch_size=32,
    img_size=224,
    is_training=False,
    num_workers=0,
    use_sampler=False
)

val_correct = defaultdict(int)
val_total = defaultdict(int)

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        for disease in diseases:
            dlabels = labels[disease].to(device)
            valid = dlabels >= 0
            if valid.sum() == 0:
                continue
            preds = torch.argmax(outputs[disease][valid], dim=1)
            gt = dlabels[valid]
            val_correct[disease] += (preds == gt).sum().item()
            val_total[disease] += valid.sum().item()

# 4. 결과 비교
print("\n4️⃣ 학습 데이터 vs 검증 데이터 성능 비교\n")
print(f"{'질환명':<20} {'학습 Acc':<12} {'검증 Acc':<12} {'차이':<10} {'판정':<15}")
print("-" * 75)

overfitting_count = 0
for disease in diseases:
    train_acc = train_correct[disease] / train_total[disease] * 100 if train_total[disease] > 0 else 0
    val_acc = val_correct[disease] / val_total[disease] * 100 if val_total[disease] > 0 else 0
    diff = train_acc - val_acc
    
    # 판정
    if diff > 10:
        status = "⚠️ 과적합 의심"
        overfitting_count += 1
    elif diff > 5:
        status = "⚡ 주의"
    else:
        status = "✅ 정상"
    
    print(f"{disease:<20} {train_acc:>6.1f}%      {val_acc:>6.1f}%      {diff:>+5.1f}%    {status}")

print("\n" + "="*75)

# 5. 종합 판정
print("\n📊 종합 분석:\n")

train_avg = sum(train_correct.values()) / sum(train_total.values()) * 100
val_avg = sum(val_correct.values()) / sum(val_total.values()) * 100
overall_diff = train_avg - val_avg

print(f"   전체 평균 정확도:")
print(f"   - 학습 데이터: {train_avg:.2f}%")
print(f"   - 검증 데이터: {val_avg:.2f}%")
print(f"   - 차이: {overall_diff:+.2f}%\n")

if overall_diff > 10:
    print("   🚨 결론: **심각한 과적합 (Severe Overfitting)**")
    print("   → 학습 데이터를 너무 외워버림")
    print("   → 새로운 데이터에서 성능 떨어질 가능성 높음\n")
    print("   💡 개선 방법:")
    print("      1. Dropout 추가/증가")
    print("      2. 데이터 증강 강화")
    print("      3. Early Stopping 더 일찍")
    print("      4. Weight Decay 증가")
    
elif overall_diff > 5:
    print("   ⚠️ 결론: **경미한 과적합 (Mild Overfitting)**")
    print("   → 약간의 과적합이 있지만 허용 범위")
    print("   → 실사용 시 성능 약간 떨어질 수 있음\n")
    print("   💡 개선 방법:")
    print("      1. 데이터 증강 추가")
    print("      2. Regularization 강화")
    
elif overall_diff < -5:
    print("   🤔 결론: **과소적합 (Underfitting)**")
    print("   → 학습이 부족하거나 모델이 너무 단순")
    print("   → 학습 더 오래 또는 모델 복잡도 증가 필요\n")
    
else:
    print("   ✅ 결론: **정상 (Good Generalization)**")
    print("   → 학습과 검증 성능이 비슷")
    print("   → 새로운 데이터에서도 비슷한 성능 기대\n")
    print("   💡 현재 상태:")
    print("      - 모델이 일반화를 잘 학습함")
    print("      - 실제 사용 준비 가능")

if overfitting_count > 3:
    print(f"\n   ⚠️ {overfitting_count}개 질환에서 과적합 의심")
    print("   → 해당 질환들은 추가 학습 또는 데이터 증강 필요")

print("\n" + "="*75 + "\n")

# 6. 데이터 누수 체크
print("5️⃣ 데이터 누수 체크...")
print("   (학습/검증 데이터에 같은 이미지가 있는지 확인)\n")

from pathlib import Path
import json

train_paths = ['../../eye_data/TL1/개/안구/일반', '../../eye_data/TL2/개/안구/일반']
val_path = '../../eye_data/VL/개/안구/일반'

train_uuids = set()
for path in train_paths:
    for json_file in Path(path).rglob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            fname = data['images']['meta']['file_name']
            # UUID 추출 (crop_ 제거)
            uuid = fname.replace('crop_', '').replace('.jpg', '').replace('.JPG', '')
            train_uuids.add(uuid)

val_uuids = set()
for json_file in Path(val_path).rglob('*.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        fname = data['images']['meta']['file_name']
        uuid = fname.replace('crop_', '').replace('.jpg', '').replace('.JPG', '')
        val_uuids.add(uuid)

overlap = train_uuids & val_uuids
overlap_ratio = len(overlap) / len(val_uuids) * 100 if len(val_uuids) > 0 else 0

print(f"   학습 데이터 UUID 수: {len(train_uuids):,}")
print(f"   검증 데이터 UUID 수: {len(val_uuids):,}")
print(f"   중복 UUID 수: {len(overlap):,} ({overlap_ratio:.1f}%)\n")

if overlap_ratio > 10:
    print("   🚨 심각한 데이터 누수!")
    print("   → 학습과 검증에 같은 이미지가 많음")
    print("   → 성능이 부풀려졌을 가능성 높음")
elif overlap_ratio > 0:
    print("   ⚠️ 약간의 데이터 누수 있음")
    print("   → 성능이 약간 부풀려졌을 수 있음")
else:
    print("   ✅ 데이터 누수 없음")
    print("   → 학습/검증 완전히 분리됨")

print("\n" + "="*75 + "\n")

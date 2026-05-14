"""
중증도 통합 빠른 테스트
기존 모델을 사용하되, 평가 시 중증도 라벨을 유/무로 통합
"""

import torch
from model import create_model
from dataset import create_dataloader
from collections import defaultdict

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("\n" + "="*70)
print("  중증도 통합 테스트 (기존 모델 활용)")
print("="*70 + "\n")

# 모델 로드
print("1️⃣ 모델 로딩...")
model = create_model('dog', pretrained=False)
ckpt = torch.load('checkpoints/dog_best.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

print(f"   ✓ 체크포인트 Epoch: {ckpt.get('epoch', 'N/A')}\n")

# 검증 데이터
print("2️⃣ 검증 데이터 로딩...")
val_loader = create_dataloader(
    data_paths=['../../eye_data/VL/개/안구/일반'],
    animal_type='dog',
    batch_size=32,
    img_size=224,
    is_training=False,
    num_workers=0,
    use_sampler=False
)

diseases = model.get_disease_names()

# 중증도 통합할 질환
SIMPLIFY_DISEASES = ['백내장', '궤양성각막질환', '비궤양성각막질환']

print(f"   중증도 통합 대상: {', '.join(SIMPLIFY_DISEASES)}\n")

# 평가
print("3️⃣ 평가 중...\n")

# 기존 방식 (중증도 구분)
original_correct = defaultdict(int)
original_total = defaultdict(int)

# 통합 방식 (유/무만)
simplified_correct = defaultdict(int)
simplified_total = defaultdict(int)

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
            
            # 기존 방식 (중증도 그대로)
            original_correct[disease] += (preds == gt).sum().item()
            original_total[disease] += valid.sum().item()
            
            # 통합 방식 (유/무만)
            if disease in SIMPLIFY_DISEASES:
                # 예측: 0(무)는 그대로, 1 이상(중증도)은 모두 1(유)로
                preds_binary = (preds > 0).long()
                gt_binary = (gt > 0).long()
                
                simplified_correct[disease] += (preds_binary == gt_binary).sum().item()
                simplified_total[disease] += valid.sum().item()
            else:
                # 다른 질환은 원래대로
                simplified_correct[disease] += (preds == gt).sum().item()
                simplified_total[disease] += valid.sum().item()

# 결과 출력
print("4️⃣ 결과 비교\n")
print(f"{'질환명':<20} {'기존 Acc':<12} {'통합 Acc':<12} {'개선':<10} {'상태'}")
print("-" * 70)

improvements = []
for disease in diseases:
    orig_acc = original_correct[disease] / original_total[disease] * 100 if original_total[disease] > 0 else 0
    simp_acc = simplified_correct[disease] / simplified_total[disease] * 100 if simplified_total[disease] > 0 else 0
    diff = simp_acc - orig_acc
    
    if disease in SIMPLIFY_DISEASES:
        status = "🎯 통합" if diff > 5 else "✅ 통합"
        improvements.append((disease, diff))
    else:
        status = "➖ 동일"
    
    print(f"{disease:<20} {orig_acc:>6.1f}%      {simp_acc:>6.1f}%      {diff:>+5.1f}%    {status}")

print("\n" + "="*70)

# 종합 분석
print("\n📊 종합 분석:\n")

orig_avg = sum(original_correct.values()) / sum(original_total.values()) * 100
simp_avg = sum(simplified_correct.values()) / sum(simplified_total.values()) * 100
overall_diff = simp_avg - orig_avg

print(f"   전체 평균 정확도:")
print(f"   - 기존 (중증도 구분): {orig_avg:.2f}%")
print(f"   - 통합 (유/무만):     {simp_avg:.2f}%")
print(f"   - 차이: {overall_diff:+.2f}%\n")

if overall_diff > 3:
    print("   ✅ 결론: **중증도 통합이 효과적!**")
    print("   → 통합 버전으로 재학습하면 성능 향상 예상")
elif overall_diff > 0:
    print("   ✅ 결론: **약간의 개선 있음**")
    print("   → 통합 버전 고려 가능")
else:
    print("   ⚠️ 결론: **통합 효과 미미**")
    print("   → 현재 모델 유지 권장")

# 개선 효과가 큰 질환
if improvements:
    print(f"\n   가장 개선된 질환:")
    for disease, diff in sorted(improvements, key=lambda x: x[1], reverse=True):
        print(f"      - {disease}: {diff:+.1f}%")

print("\n" + "="*70 + "\n")

# 다음 단계 제안
print("💡 다음 단계:")
print()
if overall_diff > 3:
    print("   1. ✅ 중증도 통합 버전으로 재학습 권장")
    print("      → 백내장, 궤양성, 비궤양성을 유/무로 변경")
    print()
    print("   2. 예상 최종 성능:")
    for disease in SIMPLIFY_DISEASES:
        new_acc = simplified_correct[disease] / simplified_total[disease] * 100
        print(f"      - {disease}: {new_acc:.1f}%")
else:
    print("   1. ⚠️ 현재 모델 유지")
    print("      → 중증도 정보가 중요하다면 현재 방식 유지")
    print()
    print("   2. 다른 개선 방법 시도:")
    print("      - 데이터 증강 강화")
    print("      - 학습 시간 연장")
    print("      - 앙상블 모델")

print("\n" + "="*70 + "\n")

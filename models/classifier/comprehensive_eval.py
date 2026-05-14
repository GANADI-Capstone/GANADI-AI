"""
종합 모델 평가 스크립트
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- 질환별 상세 분석
- 결과 저장 (JSON, CSV)
"""

import torch
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from model import create_model
from dataset import create_dataloader


def evaluate_model(model, val_loader, device, disease_names):
    """모델 평가 및 메트릭 계산"""
    model.eval()
    
    # 질환별 예측/라벨 저장
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            for disease in disease_names:
                dlabels = labels[disease].to(device)
                valid = dlabels >= 0  # -1은 라벨 없음
                
                if valid.sum() == 0:
                    continue
                
                preds = torch.argmax(outputs[disease][valid], dim=1)
                gt = dlabels[valid]
                
                predictions[disease].extend(preds.cpu().numpy())
                ground_truths[disease].extend(gt.cpu().numpy())
    
    return predictions, ground_truths


def calculate_metrics(predictions, ground_truths, disease_names):
    """질환별 메트릭 계산"""
    results = {}
    
    for disease in disease_names:
        if disease not in predictions or len(predictions[disease]) == 0:
            continue
        
        y_true = ground_truths[disease]
        y_pred = predictions[disease]
        
        # 기본 메트릭
        acc = accuracy_score(y_true, y_pred)
        
        # 이진 분류인 경우
        if len(np.unique(y_true)) == 2:
            prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
            rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        results[disease] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_samples': len(y_true),
            'unique_classes': len(np.unique(y_true))
        }
    
    return results


def print_results(results, species):
    """결과 출력"""
    print(f"\n{'='*70}")
    print(f"  {species.upper()} 모델 평가 결과")
    print(f"{'='*70}\n")
    
    # 전체 평균
    all_acc = [r['accuracy'] for r in results.values()]
    all_f1 = [r['f1_score'] for r in results.values()]
    
    print(f"전체 평균:")
    print(f"  Accuracy:  {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"  F1-Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}\n")
    
    # 질환별 상세
    print(f"{'질환명':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print("-" * 80)
    
    for disease, metrics in sorted(results.items()):
        print(f"{disease:<20} "
              f"{metrics['accuracy']:.4f}      "
              f"{metrics['precision']:.4f}      "
              f"{metrics['recall']:.4f}      "
              f"{metrics['f1_score']:.4f}      "
              f"{metrics['total_samples']:<10}")
    
    print("\n" + "="*70 + "\n")
    
    # Confusion Matrix (샘플)
    print("혼동 행렬 예시 (첫 3개 질환):")
    for disease in list(results.keys())[:3]:
        print(f"\n{disease}:")
        cm = np.array(results[disease]['confusion_matrix'])
        print(cm)


def save_results(results, species, output_dir='models/classifier/eval_results'):
    """결과 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON 저장
    json_path = output_dir / f'{species}_eval_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 결과 저장: {json_path}")
    
    # CSV 저장 (요약)
    csv_path = output_dir / f'{species}_eval_{timestamp}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('질환명,Accuracy,Precision,Recall,F1-Score,Samples\n')
        for disease, metrics in sorted(results.items()):
            f.write(f"{disease},"
                   f"{metrics['accuracy']:.4f},"
                   f"{metrics['precision']:.4f},"
                   f"{metrics['recall']:.4f},"
                   f"{metrics['f1_score']:.4f},"
                   f"{metrics['total_samples']}\n")
    
    print(f"✅ CSV 저장: {csv_path}\n")


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='강아지/고양이 모델 종합 검증')
    parser.add_argument('--species', type=str, default='dog', choices=['dog', 'cat'],
                        help='검증할 종 (dog | cat)')
    args = parser.parse_args()

    species = args.species
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"\n🔍 {species.upper()} 모델 검증 시작...")
    print(f"Device: {device}\n")
    
    # 경로 설정 (루트 기준)
    base_dir = Path(__file__).parent.parent.parent  # capstone_petcare/
    
    # 모델 로드
    model = create_model(species, pretrained=False)
    checkpoint_path = base_dir / 'models' / 'classifier' / 'checkpoints' / f'{species}_best.pth'
    
    print(f"모델 로딩: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    # 데이터 로드
    val_subpath = '개/안구/일반' if species == 'dog' else '고양이/안구/일반'
    val_path = str(base_dir / 'eye_data' / 'VL' / val_subpath)
    
    print(f"검증 데이터: {val_path}")
    val_loader = create_dataloader(
        data_paths=[val_path],
        animal_type=species,
        batch_size=32,
        img_size=224,
        is_training=False,
        num_workers=0,
        use_sampler=False
    )
    
    disease_names = model.get_disease_names()
    print(f"질환 수: {len(disease_names)}\n")
    
    # 평가
    predictions, ground_truths = evaluate_model(model, val_loader, device, disease_names)
    
    # 메트릭 계산
    results = calculate_metrics(predictions, ground_truths, disease_names)
    
    # 결과 출력
    print_results(results, species)
    
    # 결과 저장
    save_dir = base_dir / 'models' / 'classifier' / 'eval_results'
    save_results(results, species, str(save_dir))
    
    print("✅ 검증 완료!\n")


if __name__ == '__main__':
    main()

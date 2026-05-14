"""
EfficientNet-B3 멀티태스크 질환 분류 추론 스크립트
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import json

from models.classifier.model import create_model, DOG_LABEL_MAP, CAT_LABEL_MAP
from models.classifier.dataset import get_transforms


class EyeDiseasePredictor:
    """안구질환 분류 예측기"""
    
    # 중증도 통합 대상 질환 (성능 향상을 위해 유/무만 구분)
    SIMPLIFY_DISEASES = ['백내장', '궤양성각막질환', '비궤양성각막질환']
    
    def __init__(
        self,
        model_path: str,
        animal_type: str = "dog",
        device: str = None
    ):
        """
        Args:
            model_path: 학습된 모델 파일 경로 (.pth)
            animal_type: "dog" 또는 "cat"
            device: 디바이스 ("mps", "cuda", "cpu")
        """
        self.animal_type = animal_type.lower()
        
        # 디바이스 설정
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 라벨 맵
        if self.animal_type == "dog":
            self.label_map = DOG_LABEL_MAP
        else:
            self.label_map = CAT_LABEL_MAP
        
        # 역 라벨 맵 생성 (인덱스 → 라벨명)
        self.reverse_label_map = {}
        for disease, mapping in self.label_map.items():
            self.reverse_label_map[disease] = {v: k for k, v in mapping.items()}
        
        # 모델 로드
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Transform
        self.transform = get_transforms(img_size=300, is_training=False)
        
        print(f"✓ 모델 로드 완료: {model_path}")
        print(f"✓ 동물: {animal_type.upper()}")
        print(f"✓ 디바이스: {self.device}")
        print(f"✓ 중증도 통합 모드: {', '.join(self.SIMPLIFY_DISEASES)}")
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 모델 생성
        model = create_model(animal_type=self.animal_type, pretrained=False)
        
        # 가중치 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """이미지 전처리"""
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Transform 적용
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Batch 차원 추가
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        return_probabilities: bool = False
    ) -> Dict:
        """
        질환 예측
        
        Args:
            image_path: 입력 이미지 경로
            return_probabilities: 확률값 반환 여부
        
        Returns:
            {
                "predictions": {
                    "질환명": {"label": "유", "confidence": 0.91, "probabilities": [...]},
                    ...
                },
                "main_disease": "결막염",
                "main_severity": "유",
                "main_confidence": 0.91,
                "is_normal": False
            }
        """
        # 이미지 전처리
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # 추론
        outputs = self.model(image)
        
        # 결과 파싱
        predictions = {}
        max_confidence = 0.0
        main_disease = None
        main_severity = None
        
        for disease, logits in outputs.items():
            # Softmax로 확률 계산
            probs = F.softmax(logits, dim=1)[0]  # [num_classes]
            
            # 최대 확률 클래스
            pred_idx = torch.argmax(probs).item()
            pred_label = self.reverse_label_map[disease][pred_idx]
            confidence = probs[pred_idx].item()
            
            # 중증도 통합 (성능 향상)
            if disease in self.SIMPLIFY_DISEASES:
                if pred_label != "무":
                    # 모든 중증도를 "유"로 통합
                    original_label = pred_label
                    pred_label = "유"
                    # Confidence는 "무"가 아닌 모든 클래스의 확률 합
                    confidence = 1.0 - probs[0].item()  # 1 - P(무)
            
            # 저장
            result = {
                "label": pred_label,
                "confidence": confidence
            }
            
            if return_probabilities:
                result["probabilities"] = probs.cpu().tolist()
            
            predictions[disease] = result
            
            # Main disease 업데이트 (가장 높은 confidence, "무" 제외)
            if pred_label != "무" and confidence > max_confidence:
                max_confidence = confidence
                main_disease = disease
                main_severity = pred_label
        
        # 정상 여부 판단 (모든 질환이 "무"인 경우)
        is_normal = main_disease is None
        
        return {
            "predictions": predictions,
            "main_disease": main_disease,
            "main_severity": main_severity,
            "main_confidence": max_confidence,
            "is_normal": is_normal
        }
    
    def predict_batch(self, image_paths: list) -> list:
        """배치 예측"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results


def format_prediction_output(result: Dict) -> str:
    """예측 결과를 보기 좋게 포맷팅"""
    output = []
    output.append("=" * 60)
    output.append("안구질환 예측 결과")
    output.append("=" * 60)
    
    if result["is_normal"]:
        output.append("\n✓ 정상 (모든 질환 음성)")
    else:
        output.append(f"\n⚠️  주요 질환: {result['main_disease']}")
        output.append(f"   중증도: {result['main_severity']}")
        output.append(f"   신뢰도: {result['main_confidence']:.2%}")
    
    output.append("\n질환별 상세 결과:")
    output.append("-" * 60)
    
    for disease, pred in result["predictions"].items():
        label = pred["label"]
        conf = pred["confidence"]
        
        # 이모지
        emoji = "✓" if label == "무" else "⚠️"
        
        output.append(f"{emoji} {disease:20s}: {label:10s} (신뢰도: {conf:.2%})")
    
    output.append("=" * 60)
    
    return "\n".join(output)


def main():
    """CLI 테스트"""
    import argparse
    
    parser = argparse.ArgumentParser(description="안구질환 분류 추론")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="모델 파일 경로 (.pth)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="입력 이미지 경로"
    )
    parser.add_argument(
        "--animal",
        type=str,
        default="dog",
        choices=["dog", "cat"],
        help="동물 종류"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (.json)"
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="확률값 포함"
    )
    
    args = parser.parse_args()
    
    # Predictor 초기화
    predictor = EyeDiseasePredictor(
        model_path=args.model,
        animal_type=args.animal
    )
    
    # 예측
    print(f"\n🔍 예측 중: {args.image}\n")
    result = predictor.predict(
        args.image,
        return_probabilities=args.probabilities
    )
    
    # 결과 출력
    print(format_prediction_output(result))
    
    # JSON 저장 (선택사항)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 결과 저장: {args.output}")


if __name__ == "__main__":
    main()

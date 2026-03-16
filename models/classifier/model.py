"""
EfficientNet-B3 기반 멀티태스크 질환 분류 모델

구조:
- 공유 백본: EfficientNet-B3 (pretrained)
- 질환별 독립 분류 헤드
- 강아지/고양이 별도 모델
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, List


# 강아지 질환 헤드 구성 (질환명: 클래스 수)
DOG_DISEASE_HEADS = {
    "결막염": 2,           # 무/유
    "궤양성각막질환": 3,    # 무/상/하
    "백내장": 4,           # 무/초기/비성숙/성숙
    "비궤양성각막질환": 3,  # 무/상/하
    "색소침착성각막염": 2,  # 무/유
    "안검내반증": 2,        # 무/유
    "안검염": 2,            # 무/유
    "안검종양": 2,          # 무/유
    "유루증": 2,            # 무/유
    "핵경화": 2,            # 무/유
}

# 고양이 질환 헤드 구성
CAT_DISEASE_HEADS = {
    "각막궤양": 2,         # 무/유
    "각막부골편": 2,       # 무/유
    "결막염": 2,           # 무/유
    "비궤양성각막염": 2,   # 무/유
    "안검염": 2,           # 무/유
}

# 강아지 라벨 맵
DOG_LABEL_MAP = {
    "결막염": {"무": 0, "유": 1},
    "궤양성각막질환": {"무": 0, "상": 1, "하": 2},
    "백내장": {"무": 0, "초기": 1, "비성숙": 2, "성숙": 3},
    "비궤양성각막질환": {"무": 0, "상": 1, "하": 2},
    "색소침착성각막염": {"무": 0, "유": 1},
    "안검내반증": {"무": 0, "유": 1},
    "안검염": {"무": 0, "유": 1},
    "안검종양": {"무": 0, "유": 1},
    "유루증": {"무": 0, "유": 1},
    "핵경화": {"무": 0, "유": 1},
}

# 고양이 라벨 맵
CAT_LABEL_MAP = {
    "각막궤양": {"무": 0, "유": 1},
    "각막부골편": {"무": 0, "유": 1},
    "결막염": {"무": 0, "유": 1},
    "비궤양성각막염": {"무": 0, "유": 1},
    "안검염": {"무": 0, "유": 1},
}


class MultiTaskEyeDiseaseModel(nn.Module):
    """
    EfficientNet-B3 기반 멀티태스크 질환 분류 모델
    
    공유 백본 + 질환별 독립 분류 헤드
    """
    
    def __init__(self, animal_type: str = "dog", pretrained: bool = True):
        """
        Args:
            animal_type: "dog" 또는 "cat"
            pretrained: ImageNet pretrained weights 사용 여부
        """
        super().__init__()
        
        self.animal_type = animal_type.lower()
        
        if self.animal_type == "dog":
            self.disease_heads = DOG_DISEASE_HEADS
            self.label_map = DOG_LABEL_MAP
        elif self.animal_type == "cat":
            self.disease_heads = CAT_DISEASE_HEADS
            self.label_map = CAT_LABEL_MAP
        else:
            raise ValueError(f"animal_type은 'dog' 또는 'cat'이어야 합니다: {animal_type}")
        
        # EfficientNet-B3 백본 로드
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,  # 분류 헤드 제거 (feature extractor만 사용)
            global_pool='avg'
        )
        
        # 백본 출력 차원 (EfficientNet-B3: 1536)
        self.feature_dim = self.backbone.num_features
        
        # 질환별 분류 헤드 생성
        self.classifiers = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            for disease, num_classes in self.disease_heads.items()
        })
        
        print(f"✓ {animal_type.upper()} 모델 생성 완료")
        print(f"  - 백본: EfficientNet-B3 (feature_dim={self.feature_dim})")
        print(f"  - 질환 수: {len(self.disease_heads)}")
        print(f"  - 총 헤드: {sum(self.disease_heads.values())} classes")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 텐서 [B, 3, H, W]
        
        Returns:
            질환별 로짓 딕셔너리 {"질환명": 로짓 텐서 [B, num_classes]}
        """
        # 공유 백본을 통한 feature extraction
        features = self.backbone(x)  # [B, feature_dim]
        
        # 질환별 분류
        outputs = {}
        for disease, classifier in self.classifiers.items():
            outputs[disease] = classifier(features)
        
        return outputs
    
    def get_disease_names(self) -> List[str]:
        """질환 이름 리스트 반환"""
        return list(self.disease_heads.keys())
    
    def get_num_classes(self, disease: str) -> int:
        """특정 질환의 클래스 수 반환"""
        return self.disease_heads.get(disease, 0)
    
    def get_label_map(self) -> Dict:
        """라벨 맵 반환"""
        return self.label_map
    
    def freeze_backbone(self):
        """백본을 freeze하고 헤드만 학습"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ 백본 freeze 완료 (헤드만 학습)")
    
    def unfreeze_backbone(self):
        """백본을 unfreeze"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ 백본 unfreeze 완료 (전체 학습)")


def create_model(animal_type: str = "dog", pretrained: bool = True) -> MultiTaskEyeDiseaseModel:
    """
    모델 생성 헬퍼 함수
    
    Args:
        animal_type: "dog" 또는 "cat"
        pretrained: ImageNet pretrained weights 사용 여부
    
    Returns:
        MultiTaskEyeDiseaseModel 인스턴스
    """
    return MultiTaskEyeDiseaseModel(animal_type=animal_type, pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("모델 테스트")
    print("=" * 60)
    
    # 강아지 모델
    print("\n[강아지 모델]")
    dog_model = create_model("dog", pretrained=False)
    print(f"학습 가능 파라미터: {count_parameters(dog_model):,}")
    
    # 테스트 입력
    x = torch.randn(2, 3, 300, 300)  # Batch=2, 300x300 이미지
    outputs = dog_model(x)
    
    print("\n출력 형태:")
    for disease, output in outputs.items():
        print(f"  {disease:20s}: {output.shape}")
    
    # 고양이 모델
    print("\n[고양이 모델]")
    cat_model = create_model("cat", pretrained=False)
    print(f"학습 가능 파라미터: {count_parameters(cat_model):,}")
    
    outputs = cat_model(x)
    print("\n출력 형태:")
    for disease, output in outputs.items():
        print(f"  {disease:20s}: {output.shape}")

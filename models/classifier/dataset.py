"""
안구질환 분류 데이터셋 로더

데이터 경로:
- TL1: eye_data/개/안구/일반/{질환}/{중증도}/
- TL2: eye_data/TL2/개(고양이)/안구/일반/{질환}/{중증도}/
"""

import os
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# 제외할 device 유형
EXCLUDED_DEVICES = ["안구초음파", "안저카메라"]

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


class EyeDiseaseDataset(Dataset):
    """안구질환 멀티태스크 분류 데이터셋"""
    
    def __init__(
        self,
        data_paths: List[str],
        animal_type: str,
        transform=None,
        is_training: bool = True
    ):
        """
        Args:
            data_paths: 데이터 경로 리스트
            animal_type: "dog" 또는 "cat"
            transform: 이미지 변환
            is_training: 학습 모드 여부
        """
        self.data_paths = data_paths
        self.animal_type = animal_type.lower()
        self.transform = transform
        self.is_training = is_training
        
        # animal_type에 따라 label_map 자동 설정
        if self.animal_type == "dog":
            self.label_map = DOG_LABEL_MAP
        elif self.animal_type == "cat":
            self.label_map = CAT_LABEL_MAP
        else:
            raise ValueError(f"animal_type은 'dog' 또는 'cat'이어야 합니다: {animal_type}")
        
        # 질환 리스트
        self.diseases = list(self.label_map.keys())
        
        # 데이터 수집
        self.samples = self._collect_samples()
        
        print(f"\n✓ 데이터셋 생성 완료:")
        print(f"  - 동물: {animal_type.upper()}")
        print(f"  - 총 샘플 수: {len(self.samples)}")
        print(f"  - 질환 수: {len(self.diseases)}")
    
    def _collect_samples(self) -> List[Tuple[str, Dict[str, int]]]:
        """
        데이터 수집
        
        Returns:
            [(이미지 경로, {질환명: 라벨_인덱스, ...}), ...]
        """
        samples = []
        
        for data_path in self.data_paths:
            if not os.path.exists(data_path):
                print(f"⚠️  경로가 존재하지 않습니다: {data_path}")
                continue
            
            print(f"\n📁 데이터 수집 중: {data_path}")
            
            # 질환 폴더 순회
            for raw_disease_name in os.listdir(data_path):
                # macOS NFD → NFC 변환
                disease_name = unicodedata.normalize('NFC', raw_disease_name)
                
                # 라벨 맵에 있는 질환만 처리
                if disease_name not in self.label_map:
                    continue
                
                # 실제 경로는 raw_name 사용 (파일시스템이 NFD로 저장)
                disease_path = os.path.join(data_path, raw_disease_name)
                if not os.path.isdir(disease_path):
                    continue
                
                # 중증도 폴더 순회
                for raw_severity in os.listdir(disease_path):
                    # macOS NFD → NFC 변환
                    severity = unicodedata.normalize('NFC', raw_severity)
                    
                    # 라벨 맵에 있는 중증도만 처리
                    if severity not in self.label_map[disease_name]:
                        continue
                    
                    # 실제 경로는 raw_name 사용
                    severity_path = os.path.join(disease_path, raw_severity)
                    if not os.path.isdir(severity_path):
                        continue
                    
                    # JSON 파일 수집
                    for raw_filename in os.listdir(severity_path):
                        # 파일명은 NFC 변환 (확장자 체크용)
                        filename = unicodedata.normalize('NFC', raw_filename)
                        
                        if not filename.endswith('.json'):
                            continue
                        
                        # 실제 경로는 raw_name 사용
                        json_path = os.path.join(severity_path, raw_filename)
                        
                        # 이미지 파일 확인
                        img_basename = raw_filename.replace('.json', '')
                        img_path_png = os.path.join(severity_path, img_basename + '.png')
                        img_path_jpg = os.path.join(severity_path, img_basename + '.jpg')
                        
                        if os.path.exists(img_path_png):
                            img_path = img_path_png
                        elif os.path.exists(img_path_jpg):
                            img_path = img_path_jpg
                        else:
                            continue
                        
                        # JSON 데이터 읽기 및 필터링
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            label_info = data.get('label', {})
                            
                            # label_deleted == 1이면 skip
                            if label_info.get('label_deleted', 0) == 1:
                                continue
                            
                            # device 필터링
                            device = data.get('images', {}).get('meta', {}).get('device', '')
                            if device in EXCLUDED_DEVICES:
                                continue
                            
                            # label_disease_lv_3 사용 (확정 라벨)
                            disease_lv3_raw = label_info.get('label_disease_lv_3', '')
                            # JSON 내부 라벨도 NFC 변환
                            disease_lv3 = unicodedata.normalize('NFC', disease_lv3_raw)
                            if not disease_lv3:
                                continue
                            
                            # 라벨 인덱스 가져오기
                            label_idx = self.label_map[disease_name].get(disease_lv3)
                            if label_idx is None:
                                continue
                            
                            # 모든 질환에 대한 라벨 초기화 (-1: 해당 없음)
                            labels = {d: -1 for d in self.diseases}
                            labels[disease_name] = label_idx
                            
                            samples.append((img_path, labels))
                        
                        except Exception as e:
                            continue
        
        # 1차 cap — DATASET_PRIMARY_CAP (기본 50000, 사실상 무제한)
        # dataset_random_split.py의 MAX_PER_CLASS / DISEASE_CAPS가 2차 cap으로 적용됨
        MAX_PER_DISEASE_CLASS = int(os.environ.get("DATASET_PRIMARY_CAP", "50000"))
        from collections import defaultdict
        disease_class_counts = defaultdict(int)
        filtered = []
        for sample in samples:
            _, labels = sample
            # 해당 샘플의 질환과 라벨 찾기
            disease = next(d for d, v in labels.items() if v >= 0)
            label_value = labels[disease]
            # 질환+라벨 조합으로 키 생성 (예: "결막염_0", "결막염_1")
            key = f"{disease}_{label_value}"
            if disease_class_counts[key] < MAX_PER_DISEASE_CLASS:
                filtered.append(sample)
                disease_class_counts[key] += 1
        return filtered
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            image: [3, H, W] 텐서
            labels: {질환명: 라벨 텐서, ...}
        """
        img_path, labels = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Transform 적용
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # torchvision transforms
                image = self.transform(image)
        
        # 라벨을 텐서로 변환
        label_tensors = {}
        for disease in self.diseases:
            label_tensors[disease] = torch.tensor(labels[disease], dtype=torch.long)
        
        return image, label_tensors
    
    def get_class_weights(self, disease: str) -> torch.Tensor:
        """
        특정 질환의 클래스별 가중치 (빈도 역수 → Focal Loss alpha / CE weight).
        
        Args:
            disease: 질환명
        
        Returns:
            클래스 가중치 텐서 [num_classes]
        """
        # 질환별 라벨 수집
        labels = []
        for _, label_dict in self.samples:
            if label_dict[disease] >= 0:  # 유효한 라벨만
                labels.append(label_dict[disease])
        
        if not labels:
            return torch.ones(len(self.label_map[disease]))
        
        # 클래스별 샘플 수 계산
        num_classes = len(self.label_map[disease])
        class_counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
        
        # 가중치 계산 (역수)
        weights = 1.0 / (class_counts.float() + 1e-6)
        weights = weights / weights.sum() * num_classes  # 정규화
        
        return weights
    
    def get_sample_weights(
        self,
        boost_disease: Optional[str] = None,
        boost_minority_factor: float = 2.0,
    ) -> List[float]:
        """
        WeightedRandomSampler를 위한 샘플별 가중치 계산
        
        각 질환별로 클래스 불균형을 고려하여 샘플 가중치 계산.
        적은 클래스일수록 높은 가중치를 부여.

        boost_disease: 해당 질환에서 소수 클래스(minority) 샘플 가중치를 추가 배율로 곱함
          (예: 고양이 결막염 유/무 불균형 완화).
        
        Returns:
            샘플별 가중치 리스트 (len = 데이터셋 크기)
        """
        # 1. 질환별 클래스 카운트 수집
        disease_class_counts: Dict[str, Dict[int, int]] = {}

        for disease in self.diseases:
            class_counts: Dict[int, int] = defaultdict(int)

            for _, label_dict in self.samples:
                label = label_dict[disease]
                if label >= 0:
                    class_counts[label] += 1

            disease_class_counts[disease] = dict(class_counts)

        # boost 질환의 소수 클래스 라벨값
        minority_label: Optional[int] = None
        if boost_disease and boost_disease in self.diseases:
            counts = disease_class_counts.get(boost_disease, {})
            if len(counts) >= 2:
                minority_label = min(counts.keys(), key=lambda k: counts[k])

        sample_weights: List[float] = []

        for _, label_dict in self.samples:
            weights_per_disease = []

            for disease in self.diseases:
                label = label_dict[disease]

                if label >= 0:
                    count = disease_class_counts[disease].get(label, 1)
                    weight = 1.0 / count
                    weights_per_disease.append(weight)

            if weights_per_disease:
                avg_weight = sum(weights_per_disease) / len(weights_per_disease)
            else:
                avg_weight = 1.0

            if (
                minority_label is not None
                and boost_disease is not None
                and label_dict.get(boost_disease, -1) == minority_label
            ):
                avg_weight *= boost_minority_factor

            sample_weights.append(avg_weight)

        return sample_weights


def _gauss_noise_transform(p: float = 0.35):
    """Albumentations 버전별 GaussNoise 호환 (Colab 2.x / 로컬 1.x)."""
    import inspect

    params = inspect.signature(A.GaussNoise.__init__).parameters
    if "std_range" in params:
        return A.GaussNoise(std_range=(0.02, 0.08), p=p)
    return A.GaussNoise(var_limit=(15.0, 60.0), p=p)


def _training_augmentations() -> List[A.BasicTransform]:
    """강아지·고양이 공통 학습 증강 (Colab·스마트폰 촬영 환경 대응)."""
    return [
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
        A.Downscale(scale_min=0.5, scale_max=0.9, interpolation=1, p=0.3),
        _gauss_noise_transform(p=0.2),
    ]


def _strong_phone_augmentations() -> List[A.BasicTransform]:
    """하위 호환 alias — _training_augmentations 와 동일."""
    return _training_augmentations()


def get_transforms(
    img_size: int = 300,
    is_training: bool = True,
    aug_preset: str = "default",
):
    """
    이미지 변환 파이프라인
    
    Args:
        img_size: 이미지 크기 (EfficientNet-B3 권장 300)
        is_training: 학습 모드 여부
        aug_preset: \"default\" | \"strong\" | \"cat_phone\"
            strong/cat_phone — HueSaturation, BrightnessContrast, CLAHE,
            MotionBlur, ImageCompression, GaussNoise (Colab·고양이 권장)
    
    Returns:
        Albumentations Compose
    """
    norm = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if not is_training:
        return A.Compose([
            A.Resize(img_size, img_size),
            norm,
            ToTensorV2(),
        ])

    # 학습: 강아지·고양이 동일 증강 (aug_preset 은 하위 호환용, 동일 파이프라인)
    base_geo = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.35),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=0,
            p=0.25,
        ),
    ]
    return A.Compose(base_geo + _training_augmentations() + [norm, ToTensorV2()])


def create_dataloader(
    data_paths: List[str],
    animal_type: str,
    batch_size: int = 32,
    img_size: int = 300,
    is_training: bool = True,
    num_workers: int = 4,
    use_sampler: bool = False,
    aug_preset: str = "default",
    sampler_boost_disease: Optional[str] = None,
    sampler_boost_factor: float = 2.0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """
    DataLoader 생성
    
    Args:
        data_paths: 데이터 경로 리스트
        animal_type: "dog" 또는 "cat"
        batch_size: 배치 크기
        img_size: 이미지 크기
        is_training: 학습 모드 여부
        num_workers: 데이터 로더 워커 수
        use_sampler: WeightedRandomSampler 사용 여부 (클래스 불균형 처리)
        aug_preset: get_transforms 에 전달 ("default" | "strong" | "cat_phone")
        sampler_boost_disease: WeightedRandomSampler — 결막염 등 소수 클래스 부스트 질환
        sampler_boost_factor: 소수 클래스 샘플 가중 배율
        pin_memory: None 이면 False (MPS 경고 방지). CUDA 학습 시 True 권장.
    
    Returns:
        DataLoader
    """
    transform = get_transforms(img_size, is_training, aug_preset=aug_preset)

    dataset = EyeDiseaseDataset(
        data_paths=data_paths,
        animal_type=animal_type,
        transform=transform,
        is_training=is_training,
    )

    sampler = None
    shuffle = is_training

    if is_training and use_sampler:
        sample_weights = dataset.get_sample_weights(
            boost_disease=sampler_boost_disease,
            boost_minority_factor=sampler_boost_factor,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

        print(f"✓ WeightedRandomSampler 적용 (샘플 수: {len(sample_weights)})")
        if sampler_boost_disease:
            print(
                f"  · 소수 클래스 부스트: 질환={sampler_boost_disease}, 배율={sampler_boost_factor}"
            )

    use_pin = pin_memory if pin_memory is not None else False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=is_training,
    )

    return dataloader


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("데이터셋 테스트")
    print("=" * 60)
    
    # 강아지 데이터
    dog_paths = [
        "eye_data/개/안구/일반",
        "eye_data/TL2/개/안구/일반"
    ]
    
    dog_loader = create_dataloader(
        data_paths=dog_paths,
        animal_type="dog",
        batch_size=4,
        is_training=True
    )
    
    print(f"\n✓ DataLoader 생성 완료")
    print(f"  - Batch 수: {len(dog_loader)}")
    
    # 첫 번째 배치 확인
    images, labels = next(iter(dog_loader))
    print(f"\n배치 형태:")
    print(f"  - 이미지: {images.shape}")
    print(f"  - 라벨:")
    for disease, label in labels.items():
        print(f"    {disease:20s}: {label.shape}")

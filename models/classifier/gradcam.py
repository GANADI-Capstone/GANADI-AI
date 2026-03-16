"""
GradCAM을 사용한 모델 설명 가능성 시각화

사용법:
    PYTHONPATH=. python models/classifier/gradcam.py --image 이미지경로 --animal dog
    PYTHONPATH=. python models/classifier/gradcam.py --image 이미지경로 --animal cat --output gradcam_output
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.classifier.model import create_model
from models.classifier.dataset import get_transforms


class MultiTaskModelWrapper(torch.nn.Module):
    """멀티태스크 모델을 단일 출력으로 변환하는 Wrapper"""
    
    def __init__(self, model, disease_name):
        super().__init__()
        self.model = model
        self.disease_name = disease_name
    
    def forward(self, x):
        """특정 질환의 출력만 반환"""
        outputs = self.model(x)
        return outputs[self.disease_name]


class MultiTaskGradCAM:
    """멀티태스크 모델용 GradCAM Wrapper"""
    
    def __init__(self, model, target_layers, disease_name):
        """
        Args:
            model: 멀티태스크 모델
            target_layers: GradCAM을 적용할 레이어 리스트
            disease_name: 타겟 질환명
        """
        self.original_model = model
        self.disease_name = disease_name
        
        # 모델을 Wrapper로 감싸기
        self.wrapped_model = MultiTaskModelWrapper(model, disease_name)
        
        # GradCAM 생성
        self.grad_cam = GradCAM(model=self.wrapped_model, target_layers=target_layers)
    
    def __call__(self, input_tensor, target_class=None):
        """
        GradCAM 생성
        
        Args:
            input_tensor: 입력 이미지 텐서 [1, 3, H, W]
            target_class: 타겟 클래스 (None이면 predicted class 사용)
        
        Returns:
            cam: GradCAM 히트맵 [H, W]
            predicted_class: 예측 클래스
        """
        # Forward pass로 예측 클래스 얻기
        self.original_model.eval()
        with torch.no_grad():
            outputs = self.original_model(input_tensor)
            logits = outputs[self.disease_name]
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # 타겟 클래스 설정
        if target_class is None:
            target_class = predicted_class
        
        # GradCAM 생성 (targets=None이면 predicted class 사용)
        grayscale_cam = self.grad_cam(
            input_tensor=input_tensor,
            targets=None  # None이면 자동으로 predicted class 사용
        )
        
        return grayscale_cam[0], predicted_class


def load_model(animal_type: str, checkpoint_path: str = None, device: str = 'mps'):
    """
    모델 로드
    
    Args:
        animal_type: 'dog' 또는 'cat'
        checkpoint_path: 체크포인트 경로 (None이면 기본 경로 사용)
        device: 'mps', 'cuda', 또는 'cpu'
    
    Returns:
        model: 로드된 모델
        diseases: 질환 리스트
    """
    # 모델 생성
    model = create_model(animal_type=animal_type, pretrained=False)
    
    # 체크포인트 로드
    if checkpoint_path is None:
        checkpoint_path = f'models/classifier/checkpoints/{animal_type}_best.pth'
    
    print(f"체크포인트 로드: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    diseases = model.get_disease_names()
    
    return model, diseases


def preprocess_image(image_path: str, img_size: int = 224):
    """
    이미지 전처리
    
    Args:
        image_path: 이미지 경로
        img_size: 리사이즈 크기
    
    Returns:
        input_tensor: 전처리된 텐서 [1, 3, H, W]
        rgb_img: 시각화용 RGB 이미지 [H, W, 3] (0~1)
    """
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # 시각화용 이미지 (0~1)
    rgb_img = np.array(image.resize((img_size, img_size))) / 255.0
    
    # 모델 입력용 전처리
    transform = get_transforms(img_size=img_size, is_training=False)
    
    # PIL Image를 numpy로 변환
    image_np = np.array(image)
    
    # Albumentations 적용
    transformed = transform(image=image_np)
    input_tensor = transformed['image']
    
    # [C, H, W] -> [1, C, H, W]
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor, rgb_img


def visualize_gradcam(
    model,
    image_path: str,
    animal_type: str,
    output_dir: str = 'gradcam_output',
    device: str = 'mps',
    img_size: int = 224
):
    """
    GradCAM 시각화
    
    Args:
        model: 모델
        image_path: 입력 이미지 경로
        animal_type: 'dog' 또는 'cat'
        output_dir: 출력 디렉토리
        device: 디바이스
        img_size: 이미지 크기
    """
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 전처리
    input_tensor, rgb_img = preprocess_image(image_path, img_size=img_size)
    input_tensor = input_tensor.to(device)
    
    # 질환 리스트
    diseases = model.get_disease_names()
    label_map = model.get_label_map()
    
    print(f"\n이미지: {image_path}")
    print(f"동물: {animal_type.upper()}")
    print(f"질환 수: {len(diseases)}")
    print(f"\n각 질환별 GradCAM 생성 중...\n")
    
    # 각 질환별로 GradCAM 생성
    results = []
    for disease in diseases:
        # 각 질환별로 독립적인 타겟 레이어 설정
        # EfficientNet-B3의 blocks[-1]의 마지막 레이어 사용
        # 백본의 더 깊은 레이어를 사용하여 질환별 차별화
        try:
            # blocks의 마지막 MBConv block의 마지막 레이어
            target_layers = [model.backbone.blocks[-1][-1]]
        except:
            # 실패하면 conv_head 사용
            target_layers = [model.backbone.conv_head]
        
        # MultiTaskGradCAM 생성 (질환별로 독립적)
        gradcam = MultiTaskGradCAM(model, target_layers, disease)
        
        # GradCAM 생성
        cam, predicted_class = gradcam(input_tensor)
        
        # 히트맵 오버레이
        visualization = show_cam_on_image(rgb_img, cam, use_rgb=True)
        
        # 예측 결과
        class_names = list(label_map[disease].keys())
        predicted_label = class_names[predicted_class]
        
        results.append({
            'disease': disease,
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'cam': cam,
            'visualization': visualization
        })
        
        print(f"  {disease:20s} → {predicted_label}")
    
    # 결과 저장
    image_name = Path(image_path).stem
    
    # 1. 개별 히트맵 저장
    for result in results:
        disease = result['disease']
        predicted_label = result['predicted_label']
        visualization = result['visualization']
        
        output_file = output_path / f"{image_name}_{disease}_{predicted_label}.jpg"
        cv2.imwrite(str(output_file), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    # 2. 전체 그리드 저장
    n_diseases = len(diseases)
    cols = 3
    rows = (n_diseases + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n_diseases > 1 else [axes]
    
    for idx, result in enumerate(results):
        if idx < len(axes):
            disease = result['disease']
            predicted_label = result['predicted_label']
            visualization = result['visualization']
            
            axes[idx].imshow(visualization)
            axes[idx].set_title(f"{disease}\n→ {predicted_label}", fontsize=10)
            axes[idx].axis('off')
    
    # 빈 서브플롯 제거
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    grid_output = output_path / f"{image_name}_all_diseases.jpg"
    plt.savefig(grid_output, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ GradCAM 생성 완료!")
    print(f"  - 개별 히트맵: {output_path}/")
    print(f"  - 전체 그리드: {grid_output}")


def main():
    parser = argparse.ArgumentParser(description='GradCAM 시각화')
    parser.add_argument('--image', type=str, required=True, help='입력 이미지 경로')
    parser.add_argument('--animal', type=str, required=True, choices=['dog', 'cat'], help='동물 종류')
    parser.add_argument('--checkpoint', type=str, default=None, help='체크포인트 경로 (기본: models/classifier/checkpoints/{animal}_best.pth)')
    parser.add_argument('--output', type=str, default='gradcam_output', help='출력 디렉토리 (기본: gradcam_output)')
    parser.add_argument('--device', type=str, default='mps', choices=['mps', 'cuda', 'cpu'], help='디바이스')
    parser.add_argument('--img_size', type=int, default=224, help='이미지 크기')
    
    args = parser.parse_args()
    
    # 이미지 존재 확인
    if not Path(args.image).exists():
        print(f"❌ 이미지를 찾을 수 없습니다: {args.image}")
        return
    
    # 디바이스 설정
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        args.device = 'cpu'
    
    print("=" * 60)
    print("GradCAM 시각화")
    print("=" * 60)
    
    # 모델 로드
    model, diseases = load_model(args.animal, args.checkpoint, args.device)
    
    # GradCAM 생성
    visualize_gradcam(
        model=model,
        image_path=args.image,
        animal_type=args.animal,
        output_dir=args.output,
        device=args.device,
        img_size=args.img_size
    )


if __name__ == "__main__":
    main()

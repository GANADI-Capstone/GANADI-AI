"""
FastAPI AI 서버 - 반려동물 안구 질환 분석

사용법:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional, Union, List
import torch
import numpy as np
from PIL import Image
import io
import logging
import os
import sys
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import tempfile
from urllib.parse import quote

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from models.classifier.model import create_model
from models.classifier.dataset import get_transforms

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="반려동물 안구 질환 분석 API",
    description="EfficientNet-B3 기반 멀티태스크 질환 분류",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수: 모델 캐시
models_cache = {}
device = None

# Claude API 클라이언트
anthropic_client = None

# 한글 폰트 이름 (startup에서 설정)
korean_font_name = None


class DiseasePrediction(BaseModel):
    """개별 질환 예측 결과"""
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    """예측 결과 응답 모델"""
    predictions: Dict[str, DiseasePrediction]
    main_disease: str
    main_confidence: float
    is_normal: bool


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    device: str
    models_loaded: Dict[str, bool]


class ReportRequest(BaseModel):
    """리포트 생성 요청 모델"""
    animal_type: str
    pet_name: str
    predictions: Dict[str, DiseasePrediction]


class ReportResponse(BaseModel):
    """리포트 생성 응답 모델"""
    summary: str
    disease_analysis: Dict[str, str]
    visit_urgency: str
    vet_required: bool
    precautions: List[str]


class PDFReportData(BaseModel):
    """PDF 리포트 데이터 (검증용)"""
    summary: str
    disease_analysis: Dict[str, str]
    visit_urgency: str
    vet_required: bool
    precautions: List[str]


class PDFRequest(BaseModel):
    """PDF 생성 요청 모델"""
    pet_name: str
    animal_type: str
    predictions: Dict[str, DiseasePrediction]
    report: PDFReportData


def get_device():
    """최적 디바이스 선택"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(animal_type: str):
    """
    모델 로드 (캐싱)
    
    Args:
        animal_type: 'dog' 또는 'cat'
    
    Returns:
        model: 로드된 모델
    """
    global models_cache, device
    
    # 이미 로드된 모델이면 캐시에서 반환
    if animal_type in models_cache:
        return models_cache[animal_type]
    
    try:
        # 모델 생성
        model = create_model(animal_type=animal_type, pretrained=False)
        
        # 체크포인트 로드
        checkpoint_path = f'models/classifier/checkpoints/{animal_type}_best.pth'
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        logger.info(f"{animal_type.upper()} 모델 로드 중: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        
        # 캐시에 저장
        models_cache[animal_type] = model
        
        logger.info(f"✓ {animal_type.upper()} 모델 로드 완료")
        
        return model
    
    except Exception as e:
        logger.error(f"모델 로드 실패 ({animal_type}): {e}")
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {str(e)}")


def preprocess_image(image_bytes: bytes, img_size: int = 224) -> torch.Tensor:
    """
    이미지 전처리
    
    Args:
        image_bytes: 이미지 바이트
        img_size: 리사이즈 크기
    
    Returns:
        tensor: 전처리된 텐서 [1, 3, H, W]
    """
    try:
        # 이미지 로드
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # numpy 배열로 변환
        image_np = np.array(image)
        
        # Transform 적용
        transform = get_transforms(img_size=img_size, is_training=False)
        transformed = transform(image=image_np)
        tensor = transformed['image']
        
        # [C, H, W] -> [1, C, H, W]
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    except Exception as e:
        logger.error(f"이미지 전처리 실패: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 전처리 실패: {str(e)}")


def predict(model, input_tensor: torch.Tensor, animal_type: str) -> PredictionResponse:
    """
    예측 수행
    
    Args:
        model: 모델
        input_tensor: 입력 텐서 [1, 3, H, W]
        animal_type: 'dog' 또는 'cat'
    
    Returns:
        PredictionResponse: 예측 결과
    """
    try:
        # 추론
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
        
        # 질환 리스트 및 라벨 맵
        diseases = model.get_disease_names()
        label_map = model.get_label_map()
        
        # 예측 결과 파싱
        predictions = {}
        max_confidence = 0.0
        main_disease = ""
        
        for disease in diseases:
            logits = outputs[disease]
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            confidence_value = confidence.item() * 100
            predicted_idx = predicted_class.item()
            
            # 라벨 맵에서 클래스 이름 가져오기
            class_names = list(label_map[disease].keys())
            predicted_label = class_names[predicted_idx]
            
            predictions[disease] = {
                "label": predicted_label,
                "confidence": round(confidence_value, 1)
            }
            
            # "무"가 아닌 질환 중 가장 높은 confidence 찾기
            if predicted_label != "무" and confidence_value > max_confidence:
                max_confidence = confidence_value
                main_disease = disease
        
        # 모든 질환이 "무"인 경우
        is_normal = (main_disease == "")
        if is_normal:
            # 가장 높은 confidence를 가진 질환 선택 (모두 무일 때)
            for disease, pred in predictions.items():
                if pred["confidence"] > max_confidence:
                    max_confidence = pred["confidence"]
                    main_disease = disease
        
        return PredictionResponse(
            predictions=predictions,
            main_disease=main_disease,
            main_confidence=round(max_confidence, 1),
            is_normal=is_normal
        )
    
    except Exception as e:
        logger.error(f"예측 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")


def setup_korean_font():
    """한글 폰트 설정 및 등록 (macOS / Windows / Linux 공통 시도)"""
    try:
        font_paths: list[str] = []

        if sys.platform == "darwin":
            font_paths.extend(
                [
                    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                    "/Library/Fonts/NanumGothic.ttf",
                    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                ]
            )
        elif sys.platform == "win32":
            windir = os.environ.get("WINDIR", r"C:\Windows")
            fonts = Path(windir) / "Fonts"
            # ReportLab TTFont는 .ttf 위주가 안전; malgun.ttf 권장
            font_paths.extend(
                [
                    str(fonts / "malgun.ttf"),
                    str(fonts / "NanumGothic.ttf"),
                    str(fonts / "gulim.ttc"),
                ]
            )
        else:
            # Linux 등
            font_paths.extend(
                [
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                ]
            )

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont("Korean", font_path))
                    logger.info(f"✓ 한글 폰트 등록 성공: {font_path}")
                    return "Korean"
                except Exception as e:
                    logger.warning(f"폰트 등록 실패 ({font_path}): {e}")
                    continue

        logger.error("=" * 60)
        logger.error("❌ 한글 폰트를 찾지 못했습니다!")
        logger.error("=" * 60)
        logger.error("PDF 한글이 깨질 수 있습니다. OS별로 아래를 참고하세요.")
        logger.error("· macOS: 나눔고딕을 /Library/Fonts/ 에 설치")
        logger.error("· Windows: 맑은 고딕(malgun.ttf)이 보통 %WINDIR%\\Fonts 에 있습니다.")
        logger.error("· Linux: fonts-nanum 또는 Noto CJK 패키지 설치")
        logger.error("나눔고딕: https://github.com/naver/nanumfont/releases")
        logger.error("=" * 60)

        return None

    except Exception as e:
        logger.error(f"폰트 설정 중 오류 발생: {e}")
        return None


def generate_pdf(request: PDFRequest) -> str:
    """
    AI 스크리닝 결과 PDF 생성
    
    Args:
        request: PDF 생성 요청 데이터
    
    Returns:
        str: 생성된 PDF 파일 경로
    
    Raises:
        ValueError: 한글 폰트를 찾을 수 없는 경우
    """
    # 전역 폰트 이름 사용
    global korean_font_name
    
    if korean_font_name is None:
        raise ValueError(
            "한글 폰트가 설정되지 않았습니다. "
            "macOS는 /Library/Fonts/에 나눔고딕, Windows는 맑은 고딕 또는 NanumGothic.ttf를 "
            "설치한 뒤 서버를 재시작하세요. "
            "https://github.com/naver/nanumfont/releases"
        )
    
    font_name = korean_font_name
    
    # 임시 PDF 파일 생성
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_file.name
    temp_file.close()
    
    # PDF 문서 생성
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=20*mm,
        bottomMargin=20*mm,
        leftMargin=20*mm,
        rightMargin=20*mm
    )
    
    # 스토리 (PDF 내용) 리스트
    story = []
    
    # 스타일 정의
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=20,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=14,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=6,
        spaceBefore=12
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=6,
        leading=16
    )
    
    # 1. 헤더
    animal_name = "강아지" if request.animal_type == "dog" else "고양이"
    title = Paragraph(f"반려동물 AI 스크리닝 결과", title_style)
    story.append(title)
    
    date_text = Paragraph(
        f"검사일: {datetime.now().strftime('%Y년 %m월 %d일')}<br/>반려동물: {request.pet_name} ({animal_name})",
        normal_style
    )
    story.append(date_text)
    story.append(Spacer(1, 10*mm))
    
    # 2. 질환별 위험도 표
    story.append(Paragraph("질환별 위험도 평가", heading_style))
    
    # Table 데이터를 Paragraph로 감싸서 한글 폰트 보장
    from reportlab.platypus import Paragraph as P
    
    # 헤더 행
    header_style = ParagraphStyle(
        'TableHeader',
        parent=normal_style,
        fontName=font_name,
        fontSize=10,
        textColor=colors.whitesmoke,
        alignment=TA_CENTER
    )
    
    cell_style = ParagraphStyle(
        'TableCell',
        parent=normal_style,
        fontName=font_name,
        fontSize=10,
        alignment=TA_CENTER
    )
    
    table_data = [
        [P('질환명', header_style), P('판정', header_style), P('확신도', header_style)]
    ]
    
    for disease, pred in request.predictions.items():
        label = pred.label
        confidence = f"{pred.confidence}%"
        table_data.append([
            P(disease, cell_style),
            P(label, cell_style),
            P(confidence, cell_style)
        ])
    
    table = Table(table_data, colWidths=[80*mm, 30*mm, 30*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ECF0F1')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 10*mm))
    
    # 3. 종합 소견
    story.append(Paragraph("종합 소견", heading_style))
    summary_text = Paragraph(request.report.summary, normal_style)
    story.append(summary_text)
    story.append(Spacer(1, 8*mm))
    
    # 4. 질환별 상세 분석
    if request.report.disease_analysis:
        story.append(Paragraph("질환별 상세 분석", heading_style))
        for disease, analysis in request.report.disease_analysis.items():
            disease_para = Paragraph(f"<b>{disease}</b>: {analysis}", normal_style)
            story.append(disease_para)
            story.append(Spacer(1, 3*mm))
        story.append(Spacer(1, 5*mm))
    
    # 5. 수의사 방문 긴급도
    story.append(Paragraph("수의사 방문 권장", heading_style))
    urgency_color = {
        "즉시": "#E74C3C",
        "1주 이내": "#E67E22",
        "1개월 이내": "#F39C12",
        "정기검진": "#27AE60"
    }
    color = urgency_color.get(request.report.visit_urgency, "#3498DB")
    
    urgency_text = Paragraph(
        f"<font color='{color}'><b>{request.report.visit_urgency}</b></font> " +
        f"({'필수 방문' if request.report.vet_required else '선택 방문'})",
        normal_style
    )
    story.append(urgency_text)
    story.append(Spacer(1, 8*mm))
    
    # 6. 보호자 주의사항
    story.append(Paragraph("보호자 주의사항", heading_style))
    for i, precaution in enumerate(request.report.precautions, 1):
        precaution_para = Paragraph(f"{i}. {precaution}", normal_style)
        story.append(precaution_para)
        story.append(Spacer(1, 3*mm))
    
    story.append(Spacer(1, 10*mm))
    
    # 7. 면책조항
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=normal_style,
        fontSize=8,
        textColor=colors.HexColor('#7F8C8D'),
        borderWidth=1,
        borderColor=colors.HexColor('#BDC3C7'),
        borderPadding=5,
        alignment=TA_CENTER
    )
    
    disclaimer = Paragraph(
        "본 결과는 AI 스크리닝 분석 결과로, 의학적 진단이 아닙니다.<br/>" +
        "정확한 진단 및 치료를 위해서는 반드시 수의사의 진료를 받으시기 바랍니다.",
        disclaimer_style
    )
    story.append(disclaimer)
    
    # PDF 생성
    doc.build(story)
    
    return pdf_path


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 warm-up 및 폰트 설정"""
    global device, anthropic_client, korean_font_name
    
    device = get_device()
    logger.info(f"디바이스: {device}")
    
    # 한글 폰트 설정
    logger.info("한글 폰트 설정 중...")
    korean_font_name = setup_korean_font()
    if korean_font_name:
        logger.info(f"✓ PDF 생성 준비 완료 (폰트: {korean_font_name})")
    else:
        logger.warning("⚠️  한글 폰트 미설정 - PDF 생성 기능 제한됨")
    
    # Claude API 클라이언트 초기화
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        anthropic_client = Anthropic(api_key=api_key)
        logger.info("✓ Claude API 클라이언트 초기화 완료")
    else:
        logger.warning("⚠️  ANTHROPIC_API_KEY가 설정되지 않았습니다. 리포트 생성 기능을 사용할 수 없습니다.")
    
    # Dog & Cat 모델 미리 로드
    logger.info("모델 warm-up 시작...")
    
    try:
        load_model("dog")
        logger.info("✓ Dog 모델 로드 완료")
    except Exception as e:
        logger.warning(f"⚠️  Dog 모델 로드 실패: {e}")
    
    try:
        load_model("cat")
        logger.info("✓ Cat 모델 로드 완료")
    except Exception as e:
        logger.warning(f"⚠️  Cat 모델 로드 실패: {e}")
    
    logger.info("모델 warm-up 완료!")


@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "반려동물 안구 질환 분석 API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /api/ai/analyze",
            "report": "POST /api/ai/report",
            "pdf": "POST /api/ai/pdf",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        device=device,
        models_loaded={
            "dog": "dog" in models_cache,
            "cat": "cat" in models_cache
        }
    )


@app.post("/api/ai/analyze", response_model=PredictionResponse, tags=["AI"])
async def analyze(
    file: UploadFile = File(...),
    animal_type: str = Form(...)
):
    """
    안구 질환 분석 엔드포인트
    
    Args:
        file: 이미지 파일 (jpg, png)
        animal_type: 동물 종류 ('dog' 또는 'cat')
    
    Returns:
        PredictionResponse: 예측 결과
    """
    # animal_type 검증
    if animal_type not in ["dog", "cat"]:
        raise HTTPException(
            status_code=400,
            detail="animal_type은 'dog' 또는 'cat'이어야 합니다."
        )
    
    # 파일 형식 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다."
        )
    
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        
        # 모델 로드
        model = load_model(animal_type)
        
        # 이미지 전처리
        input_tensor = preprocess_image(image_bytes, img_size=224)
        
        # 예측
        result = predict(model, input_tensor, animal_type)
        
        logger.info(f"분석 완료: {animal_type} - {result.main_disease} ({result.main_confidence}%)")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")


@app.post("/api/ai/report", response_model=ReportResponse, tags=["AI"])
async def generate_report(request: ReportRequest):
    """
    AI 스크리닝 결과 리포트 생성
    
    Args:
        request: 리포트 생성 요청 (animal_type, pet_name, predictions)
    
    Returns:
        ReportResponse: 종합 소견 및 권장사항
    """
    if not anthropic_client:
        raise HTTPException(
            status_code=503,
            detail="Claude API가 설정되지 않았습니다. ANTHROPIC_API_KEY 환경변수를 확인하세요."
        )
    
    # animal_type 검증
    if request.animal_type not in ["dog", "cat"]:
        raise HTTPException(
            status_code=400,
            detail="animal_type은 'dog' 또는 'cat'이어야 합니다."
        )
    
    try:
        # predictions를 읽기 쉬운 형식으로 변환
        predictions_text = ""
        detected_diseases = []
        
        for disease, pred in request.predictions.items():
            predictions_text += f"- {disease}: {pred.label} (확신도: {pred.confidence}%)\n"
            if pred.label != "무":
                detected_diseases.append(f"{disease} ({pred.confidence}%)")
        
        animal_name = "강아지" if request.animal_type == "dog" else "고양이"
        
        # Claude API 프롬프트 구성
        prompt = f"""당신은 반려동물 안구 건강 전문가입니다. 아래 AI 스크리닝 결과를 바탕으로 보호자를 위한 소견서를 작성해주세요.

**중요 지침:**
1. 절대 "진단"이라는 표현을 사용하지 마세요. 대신 "AI 스크리닝 소견", "AI 분석 결과" 등을 사용하세요.
2. 최종 진단은 수의사만 가능하다는 점을 명확히 하세요.
3. 보호자가 이해하기 쉬운 언어로 작성하세요.
4. 객관적이고 정확한 정보를 제공하되, 과도한 불안을 조성하지 마세요.

**반려동물 정보:**
- 이름: {request.pet_name}
- 종류: {animal_name}

**AI 스크리닝 결과:**
{predictions_text}

**검출된 이상 소견:** {', '.join(detected_diseases) if detected_diseases else '없음 (정상)'}

다음 형식으로 JSON을 작성해주세요:

{{
  "summary": "전체 소견을 2-3문장으로 요약. '진단' 대신 'AI 스크리닝 소견' 표현 사용",
  "disease_analysis": {{
    "질환명1": "해당 질환의 위험도 평가 및 설명 (1-2문장)",
    "질환명2": "..."
  }},
  "visit_urgency": "즉시" 또는 "1주 이내" 또는 "1개월 이내" 또는 "정기검진",
  "vet_required": true 또는 false,
  "precautions": [
    "보호자 주의사항 1 (구체적으로)",
    "보호자 주의사항 2 (구체적으로)",
    "보호자 주의사항 3 (구체적으로)"
  ]
}}

**판단 기준:**
- 확신도 80% 이상 + "유" 검출: 즉시 또는 1주 이내, vet_required: true
- 확신도 60-80% + "유" 검출: 1개월 이내, vet_required: true
- 모두 "무": 정기검진, vet_required: false

JSON만 출력하고 다른 설명은 불필요합니다."""

        # Claude API 호출
        logger.info(f"Claude API 호출 중: {request.pet_name} ({request.animal_type})")
        
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # 응답 파싱
        response_text = message.content[0].text.strip()
        
        # JSON 파싱 (```json ``` 태그 제거)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        import json
        report_data = json.loads(response_text)
        
        # disease_analysis가 없으면 빈 dict 사용
        if "disease_analysis" not in report_data:
            report_data["disease_analysis"] = {}
        
        logger.info(f"리포트 생성 완료: {request.pet_name}")
        
        return ReportResponse(**report_data)
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}\n응답: {response_text}")
        raise HTTPException(
            status_code=500,
            detail=f"Claude API 응답 파싱 실패: {str(e)}"
        )
    except Exception as e:
        logger.error(f"리포트 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"리포트 생성 실패: {str(e)}"
        )


@app.post("/api/ai/pdf", tags=["AI"])
async def generate_pdf_report(request: PDFRequest):
    """
    AI 스크리닝 결과 PDF 생성
    
    Args:
        request: PDF 생성 요청 (pet_name, animal_type, predictions, report)
    
    Returns:
        FileResponse: PDF 파일
    """
    # animal_type 검증
    if request.animal_type not in ["dog", "cat"]:
        raise HTTPException(
            status_code=400,
            detail="animal_type은 'dog' 또는 'cat'이어야 합니다."
        )
    
    try:
        logger.info(f"PDF 생성 중: {request.pet_name} ({request.animal_type})")
        
        # PDF 생성
        pdf_path = generate_pdf(request)
        
        # 파일명 생성 (한글 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.pet_name}_AI_screening_{timestamp}.pdf"
        
        # URL 인코딩 (한글 처리)
        filename_encoded = quote(filename.encode('utf-8'))
        
        logger.info(f"PDF 생성 완료: {filename}")
        
        # PDF 파일 반환
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=filename,  # filename 파라미터는 ASCII만 허용
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{filename_encoded}"
            }
        )
    
    except Exception as e:
        logger.error(f"PDF 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF 생성 실패: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("반려동물 안구 질환 분석 API 서버")
    print("=" * 60)
    print(f"서버 주소: http://localhost:8000")
    print(f"API 문서: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 🤖 GANADI-AI

> 반려동물 안구 질환 AI 분석 서버

**GANADI (Gentle AI-assisted Natural Animal Disease Inspector)** 프로젝트의 AI 모듈입니다.

---

## 🚀 주요 기능

### 1. AI 이미지 분석
- **모델**: EfficientNet-B3 멀티태스크 학습
- **강아지**: 10개 질환 동시 분류
  - 결막염, 궤양성각막질환, 백내장, 비궤양성각막질환
  - 색소침착성각막염, 안검내반증, 안검염, 안검종양, 유루증, 핵경화
- **고양이**: 5개 질환 동시 분류
  - 결막염, 각막부골편, 각막궤양, 비궤양성각막염, 안검염

### 2. Claude AI 리포트 생성
- **자연어 소견서**: 이해하기 쉬운 한글 리포트
- **질환별 위험도 평가**: 각 질환에 대한 상세 분석
- **수의사 방문 긴급도**: 즉시/1주내/1개월내/정기검진
- **보호자 주의사항**: 3가지 구체적 조언
- **의료법 준수**: "진단" 표현 회피, "AI 스크리닝 소견" 사용

### 3. PDF 리포트 생성
- **한글 폰트 지원**: NanumGothic, AppleGothic
- **전문 레이아웃**: 병원 방문 시 활용 가능
- **포함 내용**:
  - 반려동물 정보
  - 질환별 위험도 표
  - 종합 소견
  - 수의사 방문 긴급도
  - 보호자 주의사항
  - 면책조항

### 4. GradCAM 시각화
- **질환 위치 시각화**: AI가 주목한 부위 히트맵
- **해석 가능성**: 어느 부분을 보고 판단했는지 확인

---

## 📁 프로젝트 구조

```
GANADI-AI/
├── api/                          # FastAPI AI 서버
│   ├── main.py                   # 메인 애플리케이션
│   └── test_client.py           # 테스트 클라이언트
├── models/
│   └── classifier/               # AI 모델
│       ├── train.py              # 모델 훈련
│       ├── model.py              # 모델 정의
│       ├── dataset.py            # 데이터셋 처리
│       ├── gradcam.py            # GradCAM 시각화
│       ├── predict.py            # 추론
│       ├── eval_model.py         # 평가
│       └── checkpoints/          # 모델 체크포인트 (Git 제외)
│           ├── dog_best.pth
│           └── cat_best.pth
├── docs/                         # 문서
│   ├── CLAUDE_API_GUIDE.md
│   ├── PDF_GENERATION_GUIDE.md
│   ├── KOREAN_FONT_SETUP.md
│   └── PDF_TEST_GUIDE.md
├── requirements.txt              # Python 의존성
├── .env.example                  # 환경변수 예시
├── .gitignore
└── README.md
```

---

## 🛠️ 기술 스택

### AI & ML
- **PyTorch** 2.0+ (MPS, CUDA, CPU 지원)
- **timm** (EfficientNet-B3)
- **Albumentations** (데이터 증강)
- **grad-cam** (시각화)

### API Server
- **FastAPI** 0.109.0
- **Uvicorn** (ASGI 서버)
- **httpx** (비동기 HTTP)

### AI Integration
- **Anthropic Claude** Sonnet 4 (자연어 리포트)
- **ReportLab** (PDF 생성)

---

## ⚡ 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (이미 완료)
cd GANADI-AI

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env` 파일 생성:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

### 3. 모델 파일 다운로드

> ⚠️ **필수**: 모델 파일은 용량이 커서 Git에 포함되지 않습니다.

```bash
# models/classifier/checkpoints/ 폴더에 넣기
# - dog_best.pth (~200MB)
# - cat_best.pth (~200MB)
```

**다운로드**: 팀 드라이브 또는 팀장에게 요청

### 4. 서버 실행

```bash
# AI 서버 실행 (port 8000)
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

**확인**: http://localhost:8000/docs (Swagger UI)

---

## 📡 API 엔드포인트

### `POST /api/ai/analyze`
이미지 분석 및 질환 예측

**Request**:
```bash
curl -X POST "http://localhost:8000/api/ai/analyze" \
  -F "file=@eye_image.jpg" \
  -F "animal_type=dog"
```

**Response**:
```json
{
  "predictions": {
    "결막염": {"label": "유", "confidence": 87.3},
    "백내장": {"label": "무", "confidence": 95.2},
    ...
  },
  "main_disease": "결막염",
  "main_confidence": 87.3,
  "is_normal": false
}
```

### `POST /api/ai/report`
Claude AI 소견서 생성

**Request**:
```json
{
  "animal_type": "dog",
  "pet_name": "몽이",
  "predictions": {...}
}
```

**Response**:
```json
{
  "summary": "종합 소견...",
  "disease_analysis": {"결막염": "분석..."},
  "visit_urgency": "1주 이내",
  "vet_required": true,
  "precautions": ["주의사항1", "주의사항2", "주의사항3"]
}
```

### `POST /api/ai/pdf`
PDF 리포트 생성 및 다운로드

**Response**: PDF 파일 (application/pdf)

---

## 🧪 테스트

### 테스트 클라이언트 사용

```bash
# 이미지 분석
python api/test_client.py analyze --image path/to/image.jpg --animal dog

# 리포트 생성
python api/test_client.py report \
  --pet-name 몽이 \
  --animal dog \
  --predictions '{"결막염": {"label": "유", "confidence": 87.3}}'

# PDF 생성
python api/test_client.py pdf \
  --pet-name 몽이 \
  --animal dog \
  --predictions '...' \
  --report '...'
```

---

## 🏋️ 모델 훈련

### 데이터셋 준비

```bash
# AI-Hub "153.반려동물_안구질환_데이터" 다운로드
# eye_data/ 폴더에 배치
```

### 훈련 실행

```bash
# 강아지 모델 훈련
python models/classifier/train.py --animal dog --epochs 50

# 고양이 모델 훈련
python models/classifier/train.py --animal cat --epochs 50
```

### 평가

```bash
# 모델 평가
python models/classifier/eval_model.py --animal dog
```

---

## 📊 성능 지표

### 모델 성능
- **강아지 모델**: 평균 정확도 85%+ (10개 질환)
- **고양이 모델**: 평균 정확도 80%+ (5개 질환)

### 추론 속도
- **MPS (Apple M4)**: ~200ms/image
- **CPU**: ~1-2s/image

### API 응답 시간
- `/api/ai/analyze`: ~10초 (모델 추론 포함)
- `/api/ai/report`: ~3-5초 (Claude API)
- `/api/ai/pdf`: ~2초 (PDF 생성)

---

## 📚 문서

- **[CLAUDE_API_GUIDE.md](docs/CLAUDE_API_GUIDE.md)**: Claude API 사용 가이드
- **[PDF_GENERATION_GUIDE.md](docs/PDF_GENERATION_GUIDE.md)**: PDF 생성 상세
- **[KOREAN_FONT_SETUP.md](docs/KOREAN_FONT_SETUP.md)**: 한글 폰트 설정
- **[PDF_TEST_GUIDE.md](docs/PDF_TEST_GUIDE.md)**: PDF 테스트 방법

---

## 🔧 트러블슈팅

### 모델 파일이 없음
```
FileNotFoundError: models/classifier/checkpoints/dog_best.pth
```
→ 팀 드라이브에서 모델 파일 다운로드

### 한글 폰트 오류
```
'latin-1' codec can't encode characters
```
→ [KOREAN_FONT_SETUP.md](docs/KOREAN_FONT_SETUP.md) 참고

### ANTHROPIC_API_KEY 오류
```
Error: ANTHROPIC_API_KEY not found
```
→ `.env` 파일 확인 및 API 키 설정

---

## 🤝 협업

### 브랜치 전략
```bash
# 작업 시작
git checkout -b feat/이슈번호-기능명

# 커밋
git commit -m "feat: 기능 설명"

# 푸시 및 PR
git push origin feat/이슈번호-기능명
```

### 커밋 규칙
- `feat:` 새 기능
- `fix:` 버그 수정
- `docs:` 문서 수정
- `refactor:` 리팩토링
- `test:` 테스트 추가

---

## 📞 연락

- **Organization**: https://github.com/GANADI-Capstone
- **Issues**: https://github.com/GANADI-Capstone/GANADI-AI/issues

---

## 📄 라이선스

MIT License

---

**Last Updated**: 2026.03.16  
**Version**: 1.0.0

# PDF 생성 기능 가이드

## 개요

AI 스크리닝 결과를 전문적인 PDF 리포트로 생성합니다.

## 설치

```bash
pip install reportlab>=4.0.0
```

## API 엔드포인트

### POST /api/ai/pdf

AI 스크리닝 결과 PDF 생성

#### 요청

```json
{
  "pet_name": "몽이",
  "animal_type": "dog",
  "predictions": {
    "결막염": {"label": "유", "confidence": 100},
    "백내장": {"label": "무", "confidence": 100},
    "안검염": {"label": "유", "confidence": 87.3},
    ...
  },
  "report": {
    "summary": "종합 소견 텍스트",
    "disease_analysis": {
      "결막염": "위험도 평가 텍스트",
      ...
    },
    "visit_urgency": "1주 이내",
    "vet_required": true,
    "precautions": [
      "주의사항 1",
      "주의사항 2",
      "주의사항 3"
    ]
  }
}
```

#### 응답

- **Content-Type**: `application/pdf`
- **파일명**: `{pet_name}_AI_screening_{timestamp}.pdf`

## PDF 구성

### 1. 헤더
- 제목: "반려동물 AI 스크리닝 결과"
- 검사일 (현재 날짜)
- 반려동물 이름 및 종류

### 2. 질환별 위험도 표
| 질환명 | 판정 | 확신도 |
|--------|------|--------|
| 결막염 | 유 | 100% |
| 백내장 | 무 | 100% |
| ... | ... | ... |

### 3. 종합 소견
Claude API가 생성한 전체 소견 요약

### 4. 질환별 상세 분석
검출된 질환에 대한 위험도 평가 및 설명

### 5. 수의사 방문 권장
- 긴급도: 즉시 / 1주 이내 / 1개월 이내 / 정기검진
- 색상 코딩:
  - 즉시: 빨강 (#E74C3C)
  - 1주 이내: 주황 (#E67E22)
  - 1개월 이내: 노랑 (#F39C12)
  - 정기검진: 초록 (#27AE60)

### 6. 보호자 주의사항
구체적이고 실행 가능한 조언 3가지

### 7. 면책조항
"본 결과는 AI 스크리닝 분석 결과로, 의학적 진단이 아닙니다.  
정확한 진단 및 치료를 위해서는 반드시 수의사의 진료를 받으시기 바랍니다."

## 한글 폰트

자동으로 시스템 한글 폰트를 탐색하여 사용:
1. `/System/Library/Fonts/Supplemental/AppleGothic.ttf` (macOS)
2. `/Library/Fonts/NanumGothic.ttf`
3. `/System/Library/Fonts/AppleSDGothicNeo.ttc` (macOS)

폰트를 찾지 못한 경우 기본 폰트 사용 (한글 깨질 수 있음)

## 사용 예시

### Python (requests)

```python
import requests

# 1. 이미지 분석
with open("eye.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/ai/analyze",
        files={"file": f},
        data={"animal_type": "dog"}
    )
predictions = response.json()

# 2. 리포트 생성
report_response = requests.post(
    "http://localhost:8000/api/ai/report",
    json={
        "animal_type": "dog",
        "pet_name": "몽이",
        "predictions": predictions["predictions"]
    }
)
report = report_response.json()

# 3. PDF 생성
pdf_response = requests.post(
    "http://localhost:8000/api/ai/pdf",
    json={
        "pet_name": "몽이",
        "animal_type": "dog",
        "predictions": predictions["predictions"],
        "report": report
    }
)

# 4. PDF 저장
with open("몽이_AI_screening.pdf", "wb") as f:
    f.write(pdf_response.content)

print("PDF 생성 완료!")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/ai/pdf" \
  -H "Content-Type: application/json" \
  -d @request.json \
  --output report.pdf
```

### 테스트 클라이언트

```bash
# 1. 이미지 분석 (predictions JSON 저장)
python api/test_client.py predict --image eye.jpg --animal dog

# 2. 리포트 생성 (report JSON 저장)
python api/test_client.py report \
  --pet-name 몽이 \
  --animal dog \
  --predictions eye_result.json

# 3. PDF 생성
python api/test_client.py pdf \
  --pet-name 몽이 \
  --animal dog \
  --predictions eye_result.json \
  --report 몽이_report.json
```

## 통합 워크플로우

전체 파이프라인을 한 번에 실행:

```python
import requests

def full_pipeline(image_path: str, pet_name: str, animal_type: str):
    """이미지 → 분석 → 리포트 → PDF 전체 파이프라인"""
    
    base_url = "http://localhost:8000"
    
    # 1. 이미지 분석
    print("1. 이미지 분석 중...")
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{base_url}/api/ai/analyze",
            files={"file": f},
            data={"animal_type": animal_type}
        )
    predictions = response.json()
    print(f"✓ 분석 완료: {predictions['main_disease']} ({predictions['main_confidence']}%)")
    
    # 2. Claude 리포트 생성
    print("\n2. AI 리포트 생성 중...")
    report_response = requests.post(
        f"{base_url}/api/ai/report",
        json={
            "animal_type": animal_type,
            "pet_name": pet_name,
            "predictions": predictions["predictions"]
        }
    )
    report = report_response.json()
    print(f"✓ 리포트 생성 완료")
    print(f"  긴급도: {report['visit_urgency']}")
    print(f"  수의사 방문: {'필수' if report['vet_required'] else '선택'}")
    
    # 3. PDF 생성
    print("\n3. PDF 생성 중...")
    pdf_response = requests.post(
        f"{base_url}/api/ai/pdf",
        json={
            "pet_name": pet_name,
            "animal_type": animal_type,
            "predictions": predictions["predictions"],
            "report": report
        }
    )
    
    # 4. PDF 저장
    output_file = f"{pet_name}_AI_screening.pdf"
    with open(output_file, "wb") as f:
        f.write(pdf_response.content)
    
    print(f"✓ PDF 생성 완료: {output_file}")
    print(f"  파일 크기: {len(pdf_response.content) / 1024:.1f} KB")
    
    return output_file

# 사용 예시
pdf_file = full_pipeline("eye.jpg", "몽이", "dog")
print(f"\n🎉 완료! PDF 파일: {pdf_file}")
```

## 에러 처리

### 잘못된 animal_type

```json
{
  "detail": "animal_type은 'dog' 또는 'cat'이어야 합니다."
}
```

### PDF 생성 실패

```json
{
  "detail": "PDF 생성 실패: [오류 메시지]"
}
```

## 스타일링

### 색상 팔레트

- 제목: #2C3E50 (진한 회색)
- 헤딩: #34495E (중간 회색)
- 본문: #2C3E50 (진한 회색)
- 표 헤더: #3498DB (파랑)
- 면책조항: #7F8C8D (연한 회색)

### 레이아웃

- 용지 크기: A4
- 여백: 상하좌우 20mm
- 폰트 크기:
  - 제목: 20pt
  - 헤딩: 14pt
  - 본문: 10pt
  - 면책조항: 8pt

## 파일 관리

- PDF는 임시 파일로 생성 (`tempfile.NamedTemporaryFile`)
- 다운로드 완료 후 자동으로 삭제되지 않음
- 주기적으로 임시 파일 정리 필요 (TODO)

## 향후 개선 사항

- [ ] 이미지 삽입 (원본 안구 이미지)
- [ ] GradCAM 히트맵 추가
- [ ] 차트/그래프 (질환별 확신도 바 차트)
- [ ] 다국어 지원 (영어, 일본어)
- [ ] 커스텀 테마 (병원 로고, 색상)
- [ ] 전자서명 지원
- [ ] 암호화 옵션

## 참고 자료

- [ReportLab Documentation](https://www.reportlab.com/docs/reportlab-userguide.pdf)
- [ReportLab Color Guide](https://www.reportlab.com/chartgallery/)
- [Python PDF Libraries Comparison](https://realpython.com/creating-modifying-pdf/)

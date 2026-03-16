# PDF 생성 API 테스트 가이드

## 올바른 요청 형식

### curl 명령어

```bash
curl -X 'POST' \
  'http://localhost:8000/api/ai/pdf' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "pet_name": "몽이",
  "animal_type": "dog",
  "predictions": {
    "결막염": {
      "label": "유",
      "confidence": 100
    },
    "백내장": {
      "label": "무",
      "confidence": 100
    },
    "안검염": {
      "label": "유",
      "confidence": 87.3
    },
    "유루증": {
      "label": "무",
      "confidence": 95
    },
    "핵경화": {
      "label": "무",
      "confidence": 98
    },
    "안검종양": {
      "label": "무",
      "confidence": 99
    },
    "안검내반증": {
      "label": "무",
      "confidence": 97
    },
    "궤양성각막질환": {
      "label": "무",
      "confidence": 92
    },
    "비궤양성각막질환": {
      "label": "무",
      "confidence": 88
    },
    "색소침착성각막염": {
      "label": "무",
      "confidence": 96
    }
  },
  "report": {
    "summary": "몽이의 AI 스크리닝 소견상 결막염과 안검염이 검출되었습니다. 수의사의 정확한 진단이 필요합니다.",
    "disease_analysis": {
      "결막염": "눈의 결막에 염증이 생긴 상태로 즉시 치료가 필요합니다.",
      "안검염": "눈꺼풀에 염증이 생긴 상태로 치료가 필요합니다."
    },
    "visit_urgency": "1주 이내",
    "vet_required": true,
    "precautions": [
      "눈을 비비지 않도록 엘리자베스 칼라 착용",
      "생리식염수로 눈 주변 청결 유지",
      "증상 악화 시 즉시 병원 내원"
    ]
  }
}' --output 몽이_report.pdf
```

## 일반적인 오류와 해결방법

### 422 Unprocessable Entity

#### 원인 1: predictions 객체 구조 오류

❌ **잘못된 형식:**
```json
"predictions": {
  "결막염": {"label": "유", "confidence": "100"}  // confidence가 문자열
}
```

✅ **올바른 형식:**
```json
"predictions": {
  "결막염": {"label": "유", "confidence": 100}  // confidence는 숫자
}
```

#### 원인 2: report 필드 누락

❌ **잘못된 형식:**
```json
{
  "pet_name": "몽이",
  "animal_type": "dog",
  "predictions": {...}
  // report 필드 누락
}
```

✅ **올바른 형식:**
```json
{
  "pet_name": "몽이",
  "animal_type": "dog",
  "predictions": {...},
  "report": {
    "summary": "...",
    "disease_analysis": {...},
    "visit_urgency": "1주 이내",
    "vet_required": true,
    "precautions": [...]
  }
}
```

#### 원인 3: precautions가 배열이 아님

❌ **잘못된 형식:**
```json
"report": {
  "precautions": "주의사항"  // 문자열
}
```

✅ **올바른 형식:**
```json
"report": {
  "precautions": ["주의사항1", "주의사항2", "주의사항3"]  // 배열
}
```

## Python 코드 예제

### 단계별 전체 파이프라인

```python
import requests
import json

base_url = "http://localhost:8000"

# 1. 이미지 분석
with open("eye.jpg", "rb") as f:
    analyze_response = requests.post(
        f"{base_url}/api/ai/analyze",
        files={"file": f},
        data={"animal_type": "dog"}
    )
predictions = analyze_response.json()

# 2. 리포트 생성
report_response = requests.post(
    f"{base_url}/api/ai/report",
    json={
        "animal_type": "dog",
        "pet_name": "몽이",
        "predictions": predictions["predictions"]
    }
)
report = report_response.json()

# 3. PDF 생성
pdf_request = {
    "pet_name": "몽이",
    "animal_type": "dog",
    "predictions": predictions["predictions"],
    "report": report
}

pdf_response = requests.post(
    f"{base_url}/api/ai/pdf",
    json=pdf_request
)

if pdf_response.status_code == 200:
    with open("몽이_report.pdf", "wb") as f:
        f.write(pdf_response.content)
    print("✓ PDF 생성 완료!")
else:
    print(f"❌ 오류: {pdf_response.status_code}")
    print(pdf_response.json())
```

## 스키마 검증

### PDFRequest 스키마

```python
{
  "pet_name": str,              # 필수
  "animal_type": str,           # 필수 ("dog" 또는 "cat")
  "predictions": {              # 필수
    "질환명": {
      "label": str,             # 필수 ("유" 또는 "무")
      "confidence": float       # 필수 (0-100)
    }
  },
  "report": {                   # 필수
    "summary": str,             # 필수
    "disease_analysis": dict,   # 필수 ({"질환명": "설명"})
    "visit_urgency": str,       # 필수
    "vet_required": bool,       # 필수
    "precautions": list[str]    # 필수 (최소 1개)
  }
}
```

## 디버깅

### Swagger UI에서 에러 메시지 확인

1. `/docs` 접속
2. `/api/ai/pdf` 엔드포인트 클릭
3. "Try it out" 버튼 클릭
4. 요청 데이터 입력
5. "Execute" 버튼 클릭
6. Response body에서 자세한 에러 확인

### 서버 로그 확인

```bash
# 터미널에서 서버 로그 실시간 확인
tail -f /Users/leejy/.cursor/projects/Users-leejy-capstone-petcare/terminals/2.txt
```

## 문제 해결 체크리스트

- [ ] predictions의 모든 값이 올바른 타입인가? (label: str, confidence: float)
- [ ] report 객체가 모든 필수 필드를 포함하는가?
- [ ] precautions가 배열(list)인가?
- [ ] animal_type이 "dog" 또는 "cat"인가?
- [ ] 한글 폰트가 서버에 등록되었는가? (서버 로그 확인)

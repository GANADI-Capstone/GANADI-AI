# Claude API 연동 가이드

## 개요

Claude API를 활용하여 AI 스크리닝 결과 기반 수의학적 소견서를 자동 생성합니다.

## 설치

```bash
# anthropic 패키지 설치
pip install anthropic>=0.18.0
```

## 환경 변수 설정

`.env` 파일 또는 시스템 환경변수에 API 키 설정:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

또는 `.env` 파일에 추가:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

## API 엔드포인트

### POST /api/ai/report

AI 스크리닝 결과를 바탕으로 수의학적 소견서를 생성합니다.

#### 요청

```json
{
  "animal_type": "dog",
  "pet_name": "뽀삐",
  "predictions": {
    "결막염": {
      "label": "유",
      "confidence": 87.3
    },
    "각막궤양": {
      "label": "무",
      "confidence": 92.1
    },
    "각막부골편": {
      "label": "무",
      "confidence": 95.4
    },
    "비궤양성각막염": {
      "label": "무",
      "confidence": 88.7
    },
    "안검염": {
      "label": "무",
      "confidence": 91.2
    }
  }
}
```

#### 응답

```json
{
  "summary": "뽀삐의 AI 스크리닝 결과, 결막염 소견이 87.3%의 확신도로 검출되었습니다. 다른 안구 질환은 정상 범위로 확인되었습니다.",
  "disease_analysis": {
    "결막염": "높은 확신도(87.3%)로 결막염 소견이 발견되었습니다. 결막의 충혈, 분비물, 눈곱 증가 등의 증상이 있을 수 있으며, 조기 치료가 중요합니다.",
    "각막궤양": "정상 범위입니다. 각막 표면의 손상 징후는 발견되지 않았습니다.",
    "각막부골편": "정상 범위입니다. 각막 내 색소 침착 소견은 없습니다.",
    "비궤양성각막염": "정상 범위입니다. 각막 염증 소견은 없습니다.",
    "안검염": "정상 범위입니다. 눈꺼풀 염증 소견은 없습니다."
  },
  "visit_urgency": "1주 이내",
  "vet_required": true,
  "precautions": [
    "눈을 자주 비비지 못하도록 주의하고, 필요시 보호 목걸이(엘리자베스 칼라) 착용을 고려하세요.",
    "눈 주변을 깨끗한 물이나 생리식염수로 부드럽게 닦아주고, 분비물을 제거해주세요.",
    "증상이 악화되거나 눈이 부어오르는 경우 즉시 동물병원을 방문하세요."
  ]
}
```

## 사용 예시

### Python (requests)

```python
import requests

url = "http://localhost:8000/api/ai/report"

payload = {
    "animal_type": "dog",
    "pet_name": "뽀삐",
    "predictions": {
        "결막염": {"label": "유", "confidence": 87.3},
        "각막궤양": {"label": "무", "confidence": 92.1},
        "각막부골편": {"label": "무", "confidence": 95.4},
        "비궤양성각막염": {"label": "무", "confidence": 88.7},
        "안검염": {"label": "무", "confidence": 91.2}
    }
}

response = requests.post(url, json=payload)
result = response.json()

print(f"종합 소견: {result['summary']}")
print(f"수의사 방문 긴급도: {result['visit_urgency']}")
print(f"보호자 주의사항: {result['precautions']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/ai/report" \
  -H "Content-Type: application/json" \
  -d '{
    "animal_type": "dog",
    "pet_name": "뽀삐",
    "predictions": {
      "결막염": {"label": "유", "confidence": 87.3},
      "각막궤양": {"label": "무", "confidence": 92.1},
      "각막부골편": {"label": "무", "confidence": 95.4},
      "비궤양성각막염": {"label": "무", "confidence": 88.7},
      "안검염": {"label": "무", "confidence": 91.2}
    }
  }'
```

### 테스트 클라이언트

```bash
# 1. 먼저 이미지 분석 (결과가 JSON으로 저장됨)
python api/test_client.py predict --image test.jpg --animal dog

# 2. 저장된 결과로 리포트 생성
python api/test_client.py report \
  --pet-name 뽀삐 \
  --animal dog \
  --predictions test_result.json
```

## 주요 특징

### 1. 의료법 준수
- **"진단" 표현 금지**: "AI 스크리닝 소견", "AI 분석 결과" 등의 표현 사용
- **수의사 권한 명시**: 최종 진단은 수의사만 가능하다는 점 명확히 표시
- **법적 책임 회피**: AI 분석은 참고용이며, 의료 행위가 아님을 명시

### 2. 긴급도 판단 기준

| 조건 | 긴급도 | vet_required |
|------|--------|--------------|
| 확신도 80% 이상 + "유" | 즉시 또는 1주 이내 | true |
| 확신도 60-80% + "유" | 1개월 이내 | true |
| 모두 "무" | 정기검진 | false |

### 3. 보호자 중심 언어
- 전문 용어 최소화
- 구체적이고 실행 가능한 조언
- 과도한 불안 조성 방지

## 모델 정보

- **모델**: `claude-sonnet-4-20250514`
- **Max Tokens**: 2000
- **Temperature**: 0.7 (일관성과 창의성의 균형)

## 에러 처리

### ANTHROPIC_API_KEY 미설정

```json
{
  "detail": "Claude API가 설정되지 않았습니다. ANTHROPIC_API_KEY 환경변수를 확인하세요."
}
```

**해결방법**: 환경변수에 API 키 설정

### 잘못된 animal_type

```json
{
  "detail": "animal_type은 'dog' 또는 'cat'이어야 합니다."
}
```

**해결방법**: `animal_type`을 "dog" 또는 "cat"으로 설정

### API 호출 실패

```json
{
  "detail": "리포트 생성 실패: [오류 메시지]"
}
```

**해결방법**: 
- API 키 유효성 확인
- 네트워크 연결 확인
- API 사용량 제한 확인

## 비용

Claude API 사용 비용은 사용량에 따라 부과됩니다:

- **Input**: ~$3 / million tokens
- **Output**: ~$15 / million tokens

평균 1건의 리포트 생성:
- Input: ~800 tokens
- Output: ~500 tokens
- **비용**: ~$0.01 per report

## 보안 및 개인정보

- API 키는 환경변수로 관리 (코드에 하드코딩 금지)
- 반려동물 이름 외 개인정보는 전송하지 않음
- AI 분석 결과만 전송 (원본 이미지 미전송)

## 향후 개선 사항

- [ ] 다국어 지원 (영어, 일본어)
- [ ] PDF 리포트 생성
- [ ] 이메일/SMS 발송
- [ ] 리포트 히스토리 관리
- [ ] 수의사 검토 기능

## 참고 자료

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Claude Models](https://docs.anthropic.com/en/docs/models-overview)
- [Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

# 한글 폰트 설정 가이드

## 문제 증상

PDF 생성 시 다음과 같은 오류 발생:
```
'latin-1' codec can't encode characters
```

## 원인

ReportLab의 기본 폰트(Helvetica)는 한글을 지원하지 않습니다.

## 해결 방법

### 방법 1: 나눔고딕 폰트 설치 (권장)

#### macOS

```bash
# 1. 나눔고딕 폰트 다운로드
cd ~/Downloads
curl -LO https://github.com/naver/nanumfont/releases/download/VER2.6/NanumFont_TTF_ALL.zip

# 2. 압축 해제
unzip NanumFont_TTF_ALL.zip

# 3. 시스템 폰트 폴더에 복사
sudo cp NanumFont_TTF_ALL/*.ttf /Library/Fonts/

# 4. 확인
ls /Library/Fonts/Nanum*.ttf
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get install fonts-nanum fonts-nanum-coding
```

#### Windows

1. [나눔고딕 다운로드](https://hangeul.naver.com/font)
2. 다운로드한 .ttf 파일을 더블클릭
3. "설치" 버튼 클릭

### 방법 2: 시스템 폰트 사용 (macOS)

macOS에는 AppleGothic 폰트가 기본 설치되어 있습니다:

```bash
# 폰트 확인
ls /System/Library/Fonts/Supplemental/AppleGothic.ttf
```

## 지원되는 폰트 경로

AI 서버는 다음 경로에서 자동으로 한글 폰트를 찾습니다:

1. `/System/Library/Fonts/Supplemental/AppleGothic.ttf` (macOS)
2. `/Library/Fonts/NanumGothic.ttf` (사용자 설치)
3. `/System/Library/Fonts/Supplemental/Arial Unicode.ttf` (macOS)

## 서버 로그 확인

서버 시작 시 다음과 같은 로그를 확인하세요:

### 성공 케이스
```
INFO:api.main:✓ 한글 폰트 등록 성공: /Library/Fonts/NanumGothic.ttf
```

### 실패 케이스
```
ERROR:api.main:❌ 한글 폰트를 찾지 못했습니다. PDF에 한글이 깨질 수 있습니다.
ERROR:api.main:해결 방법: 나눔고딕 폰트를 /Library/Fonts/에 설치하세요.
ERROR:api.main:다운로드: https://hangeul.naver.com/font
```

## 테스트

폰트 설치 후 PDF 생성 테스트:

```bash
# 서버 재시작 (폰트 재등록)
cd ~/capstone_petcare
source venv/bin/activate
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# 테스트
python api/test_client.py pdf \
  --pet-name 테스트 \
  --animal dog \
  --predictions test_predictions.json \
  --report test_report.json
```

생성된 PDF를 열어서 한글이 정상적으로 표시되는지 확인하세요.

## 주의사항

### .ttc 파일은 사용 불가

TrueType Collection (.ttc) 파일은 ReportLab에서 직접 사용할 수 없습니다:

❌ `/System/Library/Fonts/AppleSDGothicNeo.ttc`  
✅ `/System/Library/Fonts/Supplemental/AppleGothic.ttf`

### 권한 문제

폰트 파일에 대한 읽기 권한이 필요합니다:

```bash
# 권한 확인
ls -la /Library/Fonts/NanumGothic.ttf

# 권한 수정 (필요시)
sudo chmod 644 /Library/Fonts/NanumGothic.ttf
```

## 문제 해결

### 여전히 오류가 발생하는 경우

1. **서버 재시작**: 폰트 설치 후 반드시 서버 재시작
2. **폰트 경로 확인**: 로그에서 실제 등록된 폰트 경로 확인
3. **수동 등록**: 코드에 직접 폰트 경로 하드코딩

#### 수동 폰트 등록 (개발용)

`api/main.py`의 `setup_korean_font()` 함수 수정:

```python
def setup_korean_font():
    """한글 폰트 설정"""
    # 직접 폰트 경로 지정
    font_path = "/Library/Fonts/NanumGothic.ttf"
    
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('Korean', font_path))
        logger.info(f"✓ 한글 폰트 등록 성공: {font_path}")
        return 'Korean'
    else:
        logger.error(f"❌ 폰트 파일을 찾을 수 없습니다: {font_path}")
        return 'Helvetica'
```

## 추천 폰트

### 1. 나눔고딕 (권장)
- 용량: 약 2MB
- 특징: 가독성 좋음, 무료
- 다운로드: https://hangeul.naver.com/font

### 2. 나눔바른고딕
- 용량: 약 2.5MB
- 특징: 더 깔끔한 디자인
- 다운로드: https://hangeul.naver.com/font

### 3. AppleGothic (macOS 기본)
- 용량: 시스템 내장
- 특징: 설치 불필요, 기본적인 한글 지원

## 대안: 이미지로 변환

폰트 설치가 불가능한 환경이라면, 텍스트를 이미지로 변환하여 PDF에 삽입하는 방법도 있습니다. (권장하지 않음)

## 참고 자료

- [ReportLab 한글 폰트 가이드](https://www.reportlab.com/docs/reportlab-userguide.pdf)
- [나눔폰트 공식 사이트](https://hangeul.naver.com/font)
- [ReportLab GitHub Issues](https://github.com/MrBitBucket/reportlab-mirror/issues)

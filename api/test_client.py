"""
FastAPI 서버 테스트 클라이언트

사용법:
    python api/test_client.py --image test.jpg --animal dog
"""

import requests
import argparse
import json
from pathlib import Path


def test_health(base_url: str = "http://localhost:8000"):
    """서버 상태 확인"""
    print("=" * 60)
    print("서버 상태 확인")
    print("=" * 60)
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"\n상태: {data['status']}")
        print(f"\n모델 로드 상태:")
        for model_name, loaded in data['models'].items():
            status = "✓" if loaded else "✗"
            print(f"  {status} {model_name}: {'로드됨' if loaded else '로드 안 됨'}")
        
        print(f"\nS3: {'✓ 활성화' if data['s3'] else '✗ 비활성화'}")
        
        return True
    
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
        print("서버가 실행 중인지 확인하세요: python api/main.py")
        return False
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False


def predict_image(
    image_path: str,
    animal_type: str,
    base_url: str = "http://localhost:8000"
):
    """이미지 예측 테스트"""
    print("\n" + "=" * 60)
    print("이미지 예측")
    print("=" * 60)
    
    # 이미지 파일 확인
    if not Path(image_path).exists():
        print(f"\n❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    print(f"\n이미지: {image_path}")
    print(f"동물: {animal_type.upper()}")
    
    try:
        # 파일 업로드
        with open(image_path, 'rb') as f:
            files = {'image': (Path(image_path).name, f, 'image/jpeg')}
            data = {
                'animal_type': animal_type,
                'model_type': 'eye'
            }
            
            print(f"\n🔄 서버로 전송 중...")
            response = requests.post(
                f"{base_url}/predict",
                files=files,
                data=data
            )
        
        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 60)
            print("✅ 예측 성공!")
            print("=" * 60)
            
            # 결과 출력
            if result['is_normal']:
                print("\n✓ 정상 (모든 질환 음성)")
            else:
                print(f"\n⚠️  주요 질환: {result['main_disease']}")
                print(f"   중증도: {result['main_severity']}")
                print(f"   신뢰도: {result['main_confidence']:.2%}")
            
            print(f"\n질환별 예측 결과:")
            print("-" * 60)
            
            for disease, pred in result['predictions'].items():
                label = pred['label']
                conf = pred['confidence']
                emoji = "✓" if label == "무" else "⚠️"
                print(f"{emoji} {disease:20s}: {label:10s} (신뢰도: {conf:.2%})")
            
            # URL 출력
            if result.get('cropped_image_url'):
                print(f"\n📷 크롭 이미지: {result['cropped_image_url']}")
            
            if result.get('heatmap_url'):
                print(f"🔥 히트맵: {result['heatmap_url']}")
            
            print("\n" + "=" * 60)
            
            # JSON 저장 (선택사항)
            output_path = Path(image_path).stem + "_result.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 결과 저장: {output_path}")
        
        else:
            error = response.json()
            print("\n" + "=" * 60)
            print("❌ 예측 실패")
            print("=" * 60)
            print(f"\n에러 코드: {error['detail']['code']}")
            print(f"메시지: {error['detail']['message']}")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


def test_report(
    pet_name: str,
    animal_type: str,
    predictions_json: str,
    base_url: str = "http://localhost:8000"
):
    """리포트 생성 테스트"""
    print("=" * 60)
    print("AI 리포트 생성")
    print("=" * 60)
    
    try:
        # predictions JSON 로드
        with open(predictions_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            predictions = data.get('predictions', {})
        
        # 리포트 생성 요청
        payload = {
            "animal_type": animal_type,
            "pet_name": pet_name,
            "predictions": predictions
        }
        
        print(f"\n반려동물: {pet_name} ({animal_type})")
        print(f"예측 결과: {len(predictions)}개 질환")
        print("\n리포트 생성 중...")
        
        response = requests.post(
            f"{base_url}/api/ai/report",
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 60)
            print("📋 AI 스크리닝 소견서")
            print("=" * 60)
            
            # 종합 소견
            print(f"\n📝 종합 소견:")
            print(f"   {result['summary']}")
            
            # 질환별 분석
            print(f"\n🔍 질환별 위험도 평가:")
            print("-" * 60)
            for disease, analysis in result['disease_analysis'].items():
                print(f"\n• {disease}")
                print(f"  {analysis}")
            
            # 수의사 방문 긴급도
            urgency_emoji = {
                "즉시": "🚨",
                "1주 이내": "⚠️",
                "1개월 이내": "📅",
                "정기검진": "✓"
            }
            emoji = urgency_emoji.get(result['visit_urgency'], "📅")
            
            print(f"\n{emoji} 수의사 방문 권장: {result['visit_urgency']}")
            print(f"   {'필수' if result['vet_required'] else '선택'}")
            
            # 보호자 주의사항
            print(f"\n💡 보호자 주의사항:")
            for i, precaution in enumerate(result['precautions'], 1):
                print(f"   {i}. {precaution}")
            
            print("\n" + "=" * 60)
            
            # JSON 저장
            output_path = f"{pet_name}_report.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 리포트 저장: {output_path}")
        
        else:
            error = response.json()
            print("\n" + "=" * 60)
            print("❌ 리포트 생성 실패")
            print("=" * 60)
            print(f"\n상태 코드: {response.status_code}")
            print(f"메시지: {error.get('detail', '알 수 없는 오류')}")
    
    except FileNotFoundError:
        print(f"\n❌ 파일을 찾을 수 없습니다: {predictions_json}")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


def test_pdf(
    pet_name: str,
    animal_type: str,
    predictions_json: str,
    report_json: str,
    base_url: str = "http://localhost:8000"
):
    """PDF 생성 테스트"""
    print("=" * 60)
    print("AI PDF 리포트 생성")
    print("=" * 60)
    
    try:
        # predictions JSON 로드
        with open(predictions_json, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
            predictions = pred_data.get('predictions', {})
        
        # report JSON 로드
        with open(report_json, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # PDF 생성 요청
        payload = {
            "pet_name": pet_name,
            "animal_type": animal_type,
            "predictions": predictions,
            "report": report
        }
        
        print(f"\n반려동물: {pet_name} ({animal_type})")
        print(f"예측 결과: {len(predictions)}개 질환")
        print("\nPDF 생성 중...")
        
        response = requests.post(
            f"{base_url}/api/ai/pdf",
            json=payload
        )
        
        if response.status_code == 200:
            # PDF 파일 저장
            output_filename = f"{pet_name}_AI_screening.pdf"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print("\n" + "=" * 60)
            print("✅ PDF 생성 완료")
            print("=" * 60)
            print(f"\n📄 파일: {output_filename}")
            print(f"📦 크기: {len(response.content) / 1024:.1f} KB")
            print("\n✓ PDF 파일을 열어서 확인하세요!")
        
        else:
            error = response.json()
            print("\n" + "=" * 60)
            print("❌ PDF 생성 실패")
            print("=" * 60)
            print(f"\n상태 코드: {response.status_code}")
            print(f"메시지: {error.get('detail', '알 수 없는 오류')}")
    
    except FileNotFoundError as e:
        print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
    
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


def main():
    parser = argparse.ArgumentParser(description="FastAPI 서버 테스트 클라이언트")
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="서버 주소 (기본값: http://localhost:8000)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="명령어")
    
    # health 명령어
    health_parser = subparsers.add_parser("health", help="서버 상태 확인")
    
    # predict 명령어
    predict_parser = subparsers.add_parser("predict", help="이미지 예측")
    predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="이미지 파일 경로"
    )
    predict_parser.add_argument(
        "--animal",
        type=str,
        required=True,
        choices=["dog", "cat"],
        help="동물 종류 (dog 또는 cat)"
    )
    
    # report 명령어
    report_parser = subparsers.add_parser("report", help="AI 리포트 생성")
    report_parser.add_argument(
        "--pet-name",
        type=str,
        required=True,
        help="반려동물 이름"
    )
    report_parser.add_argument(
        "--animal",
        type=str,
        required=True,
        choices=["dog", "cat"],
        help="동물 종류 (dog 또는 cat)"
    )
    report_parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="예측 결과 JSON 파일 경로"
    )
    
    # pdf 명령어
    pdf_parser = subparsers.add_parser("pdf", help="PDF 리포트 생성")
    pdf_parser.add_argument(
        "--pet-name",
        type=str,
        required=True,
        help="반려동물 이름"
    )
    pdf_parser.add_argument(
        "--animal",
        type=str,
        required=True,
        choices=["dog", "cat"],
        help="동물 종류 (dog 또는 cat)"
    )
    pdf_parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="예측 결과 JSON 파일 경로"
    )
    pdf_parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="리포트 JSON 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 명령어가 없으면 health 체크
    if args.command is None:
        test_health(args.base_url)
        return
    
    # health 명령어
    if args.command == "health":
        test_health(args.base_url)
    
    # predict 명령어
    elif args.command == "predict":
        # 먼저 health 체크
        if test_health(args.base_url):
            predict_image(args.image, args.animal, args.base_url)
    
    # report 명령어
    elif args.command == "report":
        test_report(args.pet_name, args.animal, args.predictions, args.base_url)
    
    # pdf 명령어
    elif args.command == "pdf":
        test_pdf(args.pet_name, args.animal, args.predictions, args.report, args.base_url)


if __name__ == "__main__":
    main()

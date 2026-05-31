"""
FastAPI AI 서버 테스트 클라이언트

사용법:
    python api/test_client.py health
    python api/test_client.py predict --image test.jpg --animal dog
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def test_health(base_url: str = "http://localhost:8000") -> bool:
    print("=" * 60)
    print("서버 상태 확인")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"\n상태: {data['status']}")
        print(f"device: {data.get('device')}")
        print(f"model_version: {data.get('model_version')}")
        print(f"inference_backend: {data.get('inference_backend')}")
        print(f"checkpoint_dir: {data.get('checkpoint_dir')}")
        print("\n모델 로드:")
        for name, loaded in data.get("models_loaded", {}).items():
            mark = "✓" if loaded else "✗"
            ckpt = data.get("checkpoints", {}).get(name, "-")
            print(f"  {mark} {name}: {ckpt}")
        return True
    except requests.exceptions.ConnectionError:
        print("\n❌ 서버 연결 실패. uvicorn api.main:app --port 8000 확인")
        return False
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False


def predict_image(
    image_path: str,
    animal_type: str,
    base_url: str = "http://localhost:8000",
    device: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("이미지 분석 (/api/ai/analyze)")
    print("=" * 60)

    path = Path(image_path)
    if not path.is_file():
        print(f"\n❌ 파일 없음: {image_path}")
        return

    print(f"\n이미지: {image_path}")
    print(f"동물: {animal_type.upper()}")

    try:
        with path.open("rb") as f:
            files = {"file": (path.name, f, "image/jpeg")}
            data = {"animal_type": animal_type}
            if device:
                data["device"] = device
            response = requests.post(
                f"{base_url}/api/ai/analyze",
                files=files,
                data=data,
                timeout=120,
            )

        if response.status_code != 200:
            print(f"\n❌ HTTP {response.status_code}: {response.text}")
            return

        result = response.json()
        print("\n" + "=" * 60)
        print("✅ 분석 성공")
        print("=" * 60)
        print(f"  binary_result: {result.get('binary_result')}")
        print(f"  confidence: {result.get('confidence')}")
        print(f"  is_normal: {result.get('is_normal')}")
        print(f"  main_disease: {result.get('main_disease')}")
        print(f"  recommendation: {result.get('recommendation')}")
        if result.get("device_warning"):
            print(f"  ⚠ device_warning: {result['device_warning']}")

        print("\n  Top-3 의심 질환:")
        for i, item in enumerate(result.get("top_3_diseases", []), 1):
            print(f"    {i}. {item['disease']}: {item['confidence']:.2%}")

        print("\n  all_diseases (P 비정상):")
        for disease, prob in sorted(
            result.get("all_diseases", {}).items(),
            key=lambda x: -x[1],
        ):
            print(f"    {disease:16s} {prob:.2%}")

        out = path.stem + "_result.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 저장: {out}")

    except Exception as e:
        print(f"\n❌ 오류: {e}")


def test_report(
    pet_name: str,
    animal_type: str,
    predictions_json: str,
    base_url: str = "http://localhost:8000",
) -> None:
    with open(predictions_json, encoding="utf-8") as f:
        data = json.load(f)
    predictions = data.get("predictions", data)
    payload = {
        "animal_type": animal_type,
        "pet_name": pet_name,
        "predictions": predictions,
    }
    response = requests.post(f"{base_url}/api/ai/report", json=payload, timeout=120)
    print(response.status_code, response.text[:500] if response.text else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI API 테스트")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("health", help="헬스체크")
    p_pred = sub.add_parser("predict", help="이미지 분석")
    p_pred.add_argument("--image", required=True)
    p_pred.add_argument("--animal", default="dog", choices=["dog", "cat"])
    p_pred.add_argument("--device", default=None, help="촬영 device (스마트폰/검안경 등)")
    p_rep = sub.add_parser("report", help="리포트 생성")
    p_rep.add_argument("--pet-name", default="테스트")
    p_rep.add_argument("--animal", default="dog")
    p_rep.add_argument("--predictions", required=True)

    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    if args.command == "health" or args.command is None:
        test_health(args.base_url)
    elif args.command == "predict":
        predict_image(args.image, args.animal, args.base_url, args.device)
    elif args.command == "report":
        test_report(args.pet_name, args.animal, args.predictions, args.base_url)


if __name__ == "__main__":
    main()

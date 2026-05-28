#!/usr/bin/env bash
# 고양이 모델 — 발열 완화(BATCH=16) + cat_best.pth 이어 학습
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT="models/classifier/checkpoints/cat_best.pth"
BACKUP="models/classifier/checkpoints/cat_best.pth.bak-$(date +%Y%m%d-%H%M)"

if [[ -f "$CKPT" ]]; then
  cp "$CKPT" "$BACKUP"
  echo "📦 기존 cat_best.pth 백업: $BACKUP"
fi

if [[ -d "venv/bin" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [[ -d "backend/venv/bin" ]]; then
  echo "⚠️  ML 학습은 프로젝트 루트 venv 권장 (backend/venv 에 torch 없을 수 있음)"
  # shellcheck disable=SC1091
  source backend/venv/bin/activate
fi

echo ""
echo "🐱 고양이 학습 시작 (batch=16, MPS, cat_best 이어 학습)"
echo "   중단: Ctrl+C  |  CPU만: FORCE_CPU_TRAINING=1 $0"
echo ""

python3 models/classifier/train.py

echo ""
echo "📊 학습 후 평가:"
echo "   python3 models/classifier/comprehensive_eval.py --species cat"

#!/bin/bash

# ========================================
# KOICA TAG v3.1 - 로컬 설정 스크립트
# ========================================

set -e  # 오류 발생 시 중단

echo "🚀 KOICA TAG v3.1 로컬 환경 설정을 시작합니다..."
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Python 버전 확인
echo "1️⃣ Python 버전 확인 중..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3가 설치되어 있지 않습니다!${NC}"
    echo "Python 3.8 이상을 설치해주세요: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION 발견${NC}"
echo ""

# 2. NVIDIA GPU 확인
echo "2️⃣ NVIDIA GPU 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✅ GPU 발견: $GPU_NAME${NC}"
    echo -e "${GREEN}   VRAM: $GPU_MEMORY${NC}"
    HAS_GPU=true
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU를 찾을 수 없습니다.${NC}"
    echo -e "${YELLOW}   GPU 없이 실행하면 매우 느릴 수 있습니다.${NC}"
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    HAS_GPU=false
fi
echo ""

# 3. CUDA 버전 확인
if [ "$HAS_GPU" = true ]; then
    echo "3️⃣ CUDA 버전 확인 중..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        echo -e "${GREEN}✅ CUDA $CUDA_VERSION 발견${NC}"

        # CUDA 버전에 따라 PyTorch index 결정
        if [[ "$CUDA_VERSION" == 12.* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            LLAMA_CPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu121"
        elif [[ "$CUDA_VERSION" == 11.* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            LLAMA_CPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu118"
        else
            echo -e "${YELLOW}⚠️  지원하지 않는 CUDA 버전입니다.${NC}"
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            LLAMA_CPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu121"
        fi
    else
        echo -e "${YELLOW}⚠️  nvcc를 찾을 수 없습니다. CUDA 12.1로 진행합니다.${NC}"
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        LLAMA_CPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu121"
    fi
else
    TORCH_INDEX=""
    LLAMA_CPP_INDEX=""
fi
echo ""

# 4. 가상 환경 생성
echo "4️⃣ Python 가상 환경 생성 중..."
if [ -d "koica-env" ]; then
    echo -e "${YELLOW}⚠️  'koica-env' 디렉토리가 이미 존재합니다.${NC}"
    read -p "삭제하고 새로 만드시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf koica-env
        python3 -m venv koica-env
        echo -e "${GREEN}✅ 가상 환경 재생성 완료${NC}"
    else
        echo -e "${YELLOW}기존 가상 환경 사용${NC}"
    fi
else
    python3 -m venv koica-env
    echo -e "${GREEN}✅ 가상 환경 생성 완료${NC}"
fi
echo ""

# 5. 가상 환경 활성화
echo "5️⃣ 가상 환경 활성화 중..."
source koica-env/bin/activate
echo -e "${GREEN}✅ 가상 환경 활성화 완료${NC}"
echo ""

# 6. pip 업그레이드
echo "6️⃣ pip 업그레이드 중..."
pip install --upgrade pip setuptools wheel -q
echo -e "${GREEN}✅ pip 업그레이드 완료${NC}"
echo ""

# 7. PyTorch 설치
if [ "$HAS_GPU" = true ]; then
    echo "7️⃣ PyTorch (CUDA 지원) 설치 중... (시간이 걸릴 수 있습니다)"
    pip install torch torchvision torchaudio --index-url $TORCH_INDEX
    echo -e "${GREEN}✅ PyTorch 설치 완료${NC}"
else
    echo "7️⃣ PyTorch (CPU 전용) 설치 중... (시간이 걸릴 수 있습니다)"
    pip install torch torchvision torchaudio
    echo -e "${GREEN}✅ PyTorch 설치 완료${NC}"
fi
echo ""

# 8. PyTorch CUDA 테스트
if [ "$HAS_GPU" = true ]; then
    echo "8️⃣ PyTorch CUDA 연결 테스트 중..."
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 사용 불가'; print('✅ PyTorch CUDA 연결 성공')"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ PyTorch GPU 연결 확인${NC}"
    else
        echo -e "${RED}❌ PyTorch가 GPU를 인식하지 못했습니다.${NC}"
        echo "LOCAL_SETUP.md의 문제 해결 섹션을 참고하세요."
        exit 1
    fi
else
    echo "8️⃣ GPU 없이 진행"
fi
echo ""

# 9. llama-cpp-python 설치
if [ "$HAS_GPU" = true ]; then
    echo "9️⃣ llama-cpp-python (CUDA 지원) 설치 중..."
    pip install llama-cpp-python --extra-index-url $LLAMA_CPP_INDEX
    echo -e "${GREEN}✅ llama-cpp-python 설치 완료${NC}"
else
    echo "9️⃣ llama-cpp-python (CPU 전용) 설치 중..."
    pip install llama-cpp-python
    echo -e "${GREEN}✅ llama-cpp-python 설치 완료${NC}"
fi
echo ""

# 10. 나머지 패키지 설치
echo "🔟 나머지 패키지 설치 중..."
pip install pdfplumber gradio sentence-transformers huggingface-hub pandas numpy
echo -e "${GREEN}✅ 모든 패키지 설치 완료${NC}"
echo ""

# 11. 설치 검증
echo "1️⃣1️⃣ 설치 검증 중..."
python3 -c "
import torch
import pdfplumber
import gradio
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import pandas
import numpy

print('✅ 모든 패키지 임포트 성공')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️  GPU 가속 없음 (CPU 모드)')
"
echo ""

# 완료 메시지
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 설치가 완료되었습니다!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "다음 명령어로 프로그램을 실행하세요:"
echo ""
echo -e "${YELLOW}  source koica-env/bin/activate${NC}"
echo -e "${YELLOW}  python autotag.py${NC}"
echo ""
echo "가상 환경을 종료하려면:"
echo -e "${YELLOW}  deactivate${NC}"
echo ""
echo "자세한 사용법은 LOCAL_SETUP.md를 참고하세요."
echo ""

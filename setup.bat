@echo off
REM ========================================
REM KOICA TAG v3.1 - Windows 설정 스크립트
REM ========================================

echo 🚀 KOICA TAG v3.1 로컬 환경 설정을 시작합니다...
echo.

REM 1. Python 버전 확인
echo 1️⃣ Python 버전 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않습니다!
    echo Python 3.8 이상을 설치해주세요: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% 발견
echo.

REM 2. NVIDIA GPU 확인
echo 2️⃣ NVIDIA GPU 확인 중...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  NVIDIA GPU를 찾을 수 없습니다.
    echo GPU 없이 실행하면 매우 느릴 수 있습니다.
    set /p CONTINUE="계속하시겠습니까? (Y/N): "
    if /i not "%CONTINUE%"=="Y" exit /b 1
    set HAS_GPU=false
    set TORCH_INDEX=
    set LLAMA_CPP_INDEX=
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv,noheader') do set GPU_NAME=%%i
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=memory.total --format^=csv,noheader') do set GPU_MEMORY=%%i
    echo ✅ GPU 발견: !GPU_NAME!
    echo    VRAM: !GPU_MEMORY!
    set HAS_GPU=true

    REM CUDA 12.1 기본 사용 (Windows는 대부분 최신 CUDA 사용)
    set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    set LLAMA_CPP_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu121
)
echo.

REM 3. 가상 환경 생성
echo 4️⃣ Python 가상 환경 생성 중...
if exist koica-env (
    echo ⚠️  'koica-env' 디렉토리가 이미 존재합니다.
    set /p RECREATE="삭제하고 새로 만드시겠습니까? (Y/N): "
    if /i "%RECREATE%"=="Y" (
        rmdir /s /q koica-env
        python -m venv koica-env
        echo ✅ 가상 환경 재생성 완료
    ) else (
        echo 기존 가상 환경 사용
    )
) else (
    python -m venv koica-env
    echo ✅ 가상 환경 생성 완료
)
echo.

REM 4. 가상 환경 활성화
echo 5️⃣ 가상 환경 활성화 중...
call koica-env\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 가상 환경 활성화 실패
    pause
    exit /b 1
)
echo ✅ 가상 환경 활성화 완료
echo.

REM 5. pip 업그레이드
echo 6️⃣ pip 업그레이드 중...
python -m pip install --upgrade pip setuptools wheel -q
echo ✅ pip 업그레이드 완료
echo.

REM 6. PyTorch 설치
if "%HAS_GPU%"=="true" (
    echo 7️⃣ PyTorch (CUDA 지원) 설치 중... (시간이 걸릴 수 있습니다)
    pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
    echo ✅ PyTorch 설치 완료
) else (
    echo 7️⃣ PyTorch (CPU 전용) 설치 중... (시간이 걸릴 수 있습니다)
    pip install torch torchvision torchaudio
    echo ✅ PyTorch 설치 완료
)
echo.

REM 7. PyTorch CUDA 테스트
if "%HAS_GPU%"=="true" (
    echo 8️⃣ PyTorch CUDA 연결 테스트 중...
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA 사용 불가'; print('✅ PyTorch CUDA 연결 성공')"
    if errorlevel 1 (
        echo ❌ PyTorch가 GPU를 인식하지 못했습니다.
        echo LOCAL_SETUP.md의 문제 해결 섹션을 참고하세요.
        pause
        exit /b 1
    )
    echo ✅ PyTorch GPU 연결 확인
) else (
    echo 8️⃣ GPU 없이 진행
)
echo.

REM 8. llama-cpp-python 설치
if "%HAS_GPU%"=="true" (
    echo 9️⃣ llama-cpp-python (CUDA 지원) 설치 중...
    pip install llama-cpp-python --extra-index-url %LLAMA_CPP_INDEX%
    echo ✅ llama-cpp-python 설치 완료
) else (
    echo 9️⃣ llama-cpp-python (CPU 전용) 설치 중...
    pip install llama-cpp-python
    echo ✅ llama-cpp-python 설치 완료
)
echo.

REM 9. 나머지 패키지 설치
echo 🔟 나머지 패키지 설치 중...
pip install pdfplumber gradio sentence-transformers huggingface-hub pandas numpy
echo ✅ 모든 패키지 설치 완료
echo.

REM 10. 설치 검증
echo 1️⃣1️⃣ 설치 검증 중...
python -c "import torch; import pdfplumber; import gradio; from sentence_transformers import SentenceTransformer; from huggingface_hub import hf_hub_download; from llama_cpp import Llama; import pandas; import numpy; print('✅ 모든 패키지 임포트 성공'); print(f'✅ GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '⚠️  GPU 가속 없음 (CPU 모드)'); print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB') if torch.cuda.is_available() else None"
echo.

REM 완료 메시지
echo.
echo ========================================
echo 🎉 설치가 완료되었습니다!
echo ========================================
echo.
echo 다음 명령어로 프로그램을 실행하세요:
echo.
echo   koica-env\Scripts\activate
echo   python autotag.py
echo.
echo 가상 환경을 종료하려면:
echo   deactivate
echo.
echo 자세한 사용법은 LOCAL_SETUP.md를 참고하세요.
echo.
pause

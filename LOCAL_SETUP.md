# 🖥️ 로컬 환경 설정 가이드

KOICA TAG v3.1을 로컬 장치에서 실행하기 위한 상세 가이드입니다.

## 📋 시스템 요구사항

### 필수 사항 (Qwen2.5 32B 모델 기준)
- **GPU**: NVIDIA GPU (CUDA 지원)
  - **VRAM 16GB 이상 권장** (RTX 4090, RTX 4080, RTX 3090 또는 동급)
  - **최소 VRAM 12GB** (RTX 3060 12GB - 성능 저하 가능)
  - ✅ **노트북 GPU에서도 실행 가능**
- **CUDA**: CUDA 11.8 이상
- **Python**: Python 3.8 - 3.11 (3.12는 일부 패키지 호환성 이슈 가능)
- **RAM**: 16GB 이상 권장
- **저장 공간**: 20GB 이상 (32B 모델 크기: ~15GB)

### 운영체제
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (GPU 가속 없음, CPU 모드로만 실행 가능 - 매우 느림)

## 🚀 설치 방법

### 1단계: NVIDIA 드라이버 및 CUDA 설치

#### Windows
1. [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)
2. [CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads) (11.8 이상)
3. 설치 후 확인:
```bash
nvidia-smi
nvcc --version
```

#### Linux (Ubuntu)
```bash
# NVIDIA 드라이버 설치
sudo apt update
sudo apt install nvidia-driver-535

# CUDA 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1

# 확인
nvidia-smi
nvcc --version
```

### 2단계: Python 가상 환경 생성

```bash
# Python 3.10 사용 권장
python3 --version

# 가상 환경 생성
python3 -m venv koica-env

# 가상 환경 활성화
## Windows
koica-env\Scripts\activate

## Linux/macOS
source koica-env/bin/activate
```

### 3단계: PyTorch 설치 (CUDA 지원)

#### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### PyTorch 설치 확인
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

출력 예시:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
```

### 4단계: llama-cpp-python 설치 (CUDA 지원)

#### CUDA 12.1
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

#### CUDA 11.8
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

#### 소스에서 빌드 (선택사항, 최적 성능)
```bash
# Windows - Visual Studio Build Tools 필요
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Linux
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 5단계: 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install pdfplumber gradio sentence-transformers huggingface-hub pandas numpy
```

## 🎯 실행 방법

### 기본 실행

```bash
# 가상 환경 활성화 확인
# Windows: koica-env\Scripts\activate
# Linux/macOS: source koica-env/bin/activate

# 프로그램 실행
python autotag.py
```

### 실행 후
1. 터미널에 Gradio URL이 표시됩니다:
   ```
   Running on local URL:  http://127.0.0.1:7860
   ```

2. 웹 브라우저에서 해당 URL 접속

3. PDF 파일 업로드 및 분석 시작

## ⚠️ 문제 해결

### GPU를 인식하지 못하는 경우

**증상**: `AssertionError: ❌ GPU 런타임이 아닙니다!`

**해결 방법**:
1. NVIDIA 드라이버 설치 확인: `nvidia-smi`
2. PyTorch CUDA 설치 확인:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. CUDA 버전과 PyTorch 버전 일치 확인

### VRAM 부족 오류

**증상**: `CUDA out of memory`

**해결 방법 (Qwen2.5 32B)**:
1. **더 작은 양자화 사용**:
   - Q3_K_M (현재, ~15GB) → Q2_K (~12GB, 품질 저하)

2. `autotag.py`의 `n_ctx` 값 줄이기:
   ```python
   n_ctx=16384  # → 8192 또는 4096으로 변경
   ```

3. 다른 프로그램 종료 (크롬, 게임, IDE 등)

4. **GPU 메모리 확인**:
   ```bash
   nvidia-smi
   ```

5. **대안 모델**:
   - 더 작은 모델: Llama 3.1 8B (VRAM 8GB)
   - 더 큰 모델: Llama 3.1 70B (VRAM 40GB 필요, 클라우드 권장)

### llama-cpp-python 설치 오류

**Windows 사용자**:
1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) 설치
2. "Desktop development with C++" 워크로드 선택
3. 재부팅 후 다시 설치

**Linux 사용자**:
```bash
sudo apt install build-essential cmake
```

### 모델 다운로드 느림

**해결 방법**:
1. Hugging Face 계정 생성 및 토큰 발급
2. 환경 변수 설정:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

## 🔧 고급 설정

### 모델 경로 변경

`autotag.py`에서 모델 저장 위치 변경:
```python
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct-GGUF",
    filename="qwen2.5-32b-instruct-q3_k_m.gguf",
    local_dir="/your/custom/path/models"  # 여기를 변경
)
```

### 성능 튜닝

`autotag.py`의 LLM 초기화 부분 수정:
```python
llm = Llama(
    model_path=model_path,
    n_ctx=16384,        # 컨텍스트 길이 (메모리 ↔ 성능)
    n_gpu_layers=-1,    # -1 = 전체 GPU 사용
    n_batch=512,        # 배치 크기 증가 시 속도 향상
    n_threads=4,        # CPU 스레드 수
    use_mlock=True,     # RAM 고정 (빠름)
    verbose=False
)
```

## 📊 성능 벤치마크

### Qwen2.5 32B (Q3_K_M) - 현재 버전
| GPU | VRAM | 지원 여부 | 처리 시간 (30페이지 PDF) |
|-----|------|----------|------------------------|
| RTX 4090 Desktop | 24GB | ✅ | ~4-5분 |
| RTX 4090 Laptop | 16GB | ✅ | ~5-7분 |
| RTX 4080 Laptop | 12GB | ✅ | ~6-8분 |
| RTX 3090 | 24GB | ✅ | ~5-7분 |
| RTX 3080 | 10GB | ❌ | 메모리 부족 |
| RTX 3060 12GB | 12GB | ⚠️ | ~8-12분 |

### 다른 모델 비교
| 모델 | VRAM 요구 | 성능 | 실행 가능 GPU |
|------|----------|------|-------------|
| Llama 3.1 8B | 8GB | ⭐⭐⭐ | RTX 3060+ |
| **Qwen2.5 32B** | **15GB** | **⭐⭐⭐⭐** | **RTX 4090 Laptop+** |
| Llama 3.1 70B | 40GB | ⭐⭐⭐⭐⭐ | A100, H100 |

✅ **32B 모델 권장**: 노트북에서 실행 가능하면서 우수한 성능

## 📞 지원

- GitHub Issues: [프로젝트 이슈](https://github.com/amnotyoung/autotag/issues)
- 문서: README.md

## ✅ 체크리스트

설치 완료 전 확인:
- [ ] NVIDIA 드라이버 설치됨 (`nvidia-smi` 작동)
- [ ] CUDA 설치됨 (`nvcc --version` 작동)
- [ ] Python 가상 환경 생성 및 활성화
- [ ] PyTorch CUDA 버전 설치 (`torch.cuda.is_available() == True`)
- [ ] llama-cpp-python CUDA 버전 설치
- [ ] requirements.txt 패키지 설치
- [ ] `python autotag.py` 실행 시 GPU 인식 확인

모든 체크박스를 확인했다면 준비 완료입니다! 🎉

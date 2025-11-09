# 🚀 빠른 시작 가이드

Qwen2.5 32B 모델을 로컬에서 테스트하기 위한 단계별 가이드입니다.

## ⚠️ 먼저 확인하세요!

### GPU 확인
```bash
nvidia-smi
```

**필요한 GPU VRAM**:
- ✅ **16GB 이상**: RTX 4090 Laptop, RTX 4080, RTX 3090 → 완벽하게 실행 가능
- ⚠️ **12GB**: RTX 3060 12GB, RTX 4060 Ti → 실행 가능 (약간 느림)
- ❌ **12GB 미만**: RTX 3060 8GB, RTX 3070 → **32B 모델 실행 불가** (8B 모델 권장)

---

## 📦 1단계: 환경 설정

### Python 가상 환경 생성
```bash
# 프로젝트 폴더로 이동
cd autotag-onprem

# 가상 환경 생성
python3 -m venv koica-env

# 활성화
# Windows:
koica-env\Scripts\activate

# Linux/macOS:
source koica-env/bin/activate
```

### PyTorch 설치 (CUDA 지원)
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### GPU 인식 확인
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

**출력 예시**:
```
CUDA: True
GPU: NVIDIA A100-SXM4-40GB
VRAM: 40.0GB
```

---

## 🔧 2단계: 패키지 설치

### llama-cpp-python (CUDA 지원)
```bash
# CUDA 12.1
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# CUDA 11.8
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

### 나머지 패키지
```bash
pip install pdfplumber gradio sentence-transformers huggingface-hub pandas numpy
```

---

## 🎯 3단계: 실행

```bash
python autotag.py
```

### 첫 실행 시 일어나는 일
1. **모델 다운로드** (~15GB, 5-10분 소요)
   ```
   📥 Qwen2.5 32B 다운로드 중...
   Downloading...  [████████] 100%
   ```

2. **GPU 메모리 할당**
   ```
   ✅ GPU: NVIDIA GeForce RTX 4090 Laptop
   ✅ VRAM: 16.0GB
   🔄 LLM 초기화 중...
   ✅ LLM 준비 완료!
   ```

3. **Gradio 인터페이스 시작**
   ```
   Running on local URL:  http://127.0.0.1:7860
   ```

### 브라우저에서 접속
1. 웹 브라우저 열기
2. `http://127.0.0.1:7860` 접속
3. PDF 파일 업로드
4. 분석 시작 (30페이지 기준 5-10분)

---

## 🐛 문제 해결

### 문제 1: "GPU 런타임이 아닙니다!"
```bash
# PyTorch CUDA 설치 확인
python -c "import torch; print(torch.cuda.is_available())"

# False가 나오면 PyTorch 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 문제 2: "CUDA out of memory"

**12GB GPU 사용자**:
`autotag.py`에서 더 작은 양자화 모델로 변경:
```python
# 40-42번째 줄 수정
model_path = hf_hub_download(
    repo_id="bartowski/Qwen2.5-32B-Instruct-GGUF",
    filename="Qwen2.5-32B-Instruct-Q2_K.gguf",  # Q3_K_M → Q2_K
    local_dir="./models"
)
```

**그래도 안 되면 컨텍스트 크기 줄이기**:
```python
# 47-55번째 줄 수정
llm = Llama(
    model_path=model_path,
    n_ctx=8192,  # 16384 → 8192로 변경
    n_gpu_layers=-1,
    n_batch=512,
    n_threads=4,
    use_mlock=True,
    verbose=False
)
```

### 문제 3: 12GB 미만 GPU
32B 모델 대신 8B 모델 사용:
```python
# autotag.py 40-42번째 줄
model_path = hf_hub_download(
    repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
    local_dir="./models"
)
```

---

## 📊 실행 확인

### 정상 실행 시 터미널 출력 예시:
```
✅ GPU: NVIDIA A100-SXM4-40GB
✅ VRAM: 40.0GB

✅ GPU 확인 완료! 패키지가 설치되어 있는지 확인 중...

📥 Llama 3.1 70B 다운로드 중...
🔄 LLM 초기화 중...
✅ LLM 준비 완료!

🔄 한국어 임베딩 모델 로딩...
✅ 한국어 임베딩 준비 완료!

================================================================================
🚀 KOICA TAG v3.1 (예시 복사 방지 강화)
================================================================================

🔧 v3.1 개선:
  - 예시를 형식 가이드로 변경 (구체적 내용 제거)
  - 예시 내용 복사 절대 금지 명시
  - 예시 복사 검증 로직 추가
  - 실제 문서 내용만 사용 강조

================================================================================
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

---

## ✅ 체크리스트

실행 전 확인:
- [ ] GPU VRAM 16GB 이상 (또는 12GB + Q2_K 양자화)
- [ ] NVIDIA 드라이버 설치됨 (`nvidia-smi` 작동)
- [ ] Python 가상 환경 활성화
- [ ] PyTorch CUDA 설치 (`torch.cuda.is_available() == True`)
- [ ] llama-cpp-python 설치
- [ ] 저장 공간 20GB 이상 확보

모든 항목 확인 완료 → `python autotag.py` 실행!

---

## 💡 팁

### 더 빠른 다운로드
Hugging Face 토큰 사용:
```bash
export HF_TOKEN="your_token_here"
python autotag.py
```

### 모델 저장 위치 확인
```bash
ls -lh models/
```

### GPU 메모리 사용량 모니터링
터미널 새 창에서:
```bash
watch -n 1 nvidia-smi
```

---

## 📞 도움이 필요하면

1. `LOCAL_SETUP.md` - 상세 설치 가이드
2. `README.md` - 프로젝트 전체 문서
3. GitHub Issues - 문제 리포트

🎉 준비되셨으면 시작하세요!

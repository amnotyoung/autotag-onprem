# 🎯 KOICA TAG v4.0

KOICA 사업 문서 자동 검토 시스템 (Technical Appraisal Guide)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amnotyoung/autotag-onprem/blob/claude/test-google-colab-011CUzGGNyvLiNjMy5Utmg6V/KOICA_TAG_v4.0_Colab.ipynb)

## 🚀 빠른 시작

### Colab에서 바로 실행 (권장)

1. 위 "Open in Colab" 버튼 클릭
2. GPU 런타임 설정: `Runtime → Change runtime type → GPU`
3. 셀 실행 (Shift + Enter)
4. PDF 업로드하고 분석 시작!

### 로컬에서 실행

#### 자동 설치 (권장)

**Linux/macOS**:
```bash
chmod +x setup.sh
./setup.sh
```

**Windows**:
```bash
setup.bat
```

#### 수동 설치

1. **환경 설정**:
   ```bash
   # 가상 환경 생성
   python3 -m venv koica-env

   # 가상 환경 활성화
   ## Linux/macOS:
   source koica-env/bin/activate
   ## Windows:
   koica-env\Scripts\activate
   ```

2. **패키지 설치**:
   ```bash
   # PyTorch (CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # llama-cpp-python (CUDA 지원)
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

   # 나머지 패키지
   pip install -r requirements.txt
   ```

3. **실행**:
   ```bash
   python autotag.py
   ```

📘 **상세 가이드**: [LOCAL_SETUP.md](LOCAL_SETUP.md) 참고

## 🔧 v4.0 핵심 개선

1. ✅ **PMC Agent 제거**: LLM 호출 6회 → 1회로 대폭 축소
2. ✅ **섹터 전문가 집중**: 섹터별 핵심 이슈 + 필수 질문 심층 검토
3. ✅ **처리 속도 향상**: Agent 부담 감소로 약 5~6배 빠름
4. ✅ **검토 품질 강화**: 섹터 전문성에 집중한 분석
5. ✅ **Qwen2.5 32B**: 최신 모델로 우수한 성능 + 빠른 속도 (A100 최적)

## 💡 해결된 문제

| 문제 | 해결 |
|------|------|
| ❌ 복잡한 Multi-Agent | ✅ 단일 섹터 전문가 집중 분석 |
| ❌ 느린 처리 속도 (5-10분) | ✅ 5~6배 빠른 분석 (2-4분) |
| ❌ 작은 모델의 한계 | ✅ Qwen2.5 32B로 품질 향상 |

## 📊 기능

- **섹터 전문가 집중 분석**: 섹터별 핵심 이슈 심층 검토
- **12개 섹터 지원**: 교육, 보건, 거버넌스, 농촌개발, 물, 에너지, 교통, 도시, 과학기술, 기후행동, 성평등, 인권
- **RAG 기반**: 문서 내 실제 내용 검색 및 인용
- **검증 로직**: 플레이스홀더, 논리적 모순, 예시 복사 자동 검출

## 📋 시스템 요구사항

- **GPU**: NVIDIA GPU (Colab A100 권장, T4 가능)
- **VRAM**: 20GB 이상 권장 (Qwen2.5 32B Q4)
- **Python**: 3.8+
- **Colab**: Pro/Pro+ 권장 (무료 T4도 가능하나 느릴 수 있음)

## 📖 사용 예시

1. PDF 업로드
2. 자동 분석 (2-4분, v3.1 대비 5~6배 빠름)
3. 결과 확인:
   - 0️⃣ 문서 정보
   - 1️⃣ 사업 핵심 정보
   - 2️⃣ 섹터 전문가 심층 분석
   - 3️⃣ 통합 권고안
4. TXT/HTML 다운로드

## 🔄 버전 히스토리

- **v4.0** (2025-01-XX): 섹터 전문가 집중, 처리 속도 5~6배 향상
- **v3.1**: 예시 복사 방지 강화
- **v3.0**: 질문 생성 자동화
- **v2.9**: Multi-Agent 시스템 도입

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR 환영합니다!

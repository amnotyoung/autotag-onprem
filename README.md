# 🎯 KOICA TAG v3.1

KOICA 사업 문서 자동 검토 시스템 (Technical Appraisal Guide)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amnotyoung/autotag/blob/claude/fix-solar-analysis-rainy-season-011CUv6NG6frc4X4LBy2vEBd/KOICA_TAG_v3.1.ipynb)

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

## 🔧 v3.1 핵심 개선

1. ✅ **예시 복사 방지**: 형식 가이드로 변경, 구체적 내용 제거
2. ✅ **명시적 지시 강화**: "태양광, 디젤 등 예시 내용 복사 금지"
3. ✅ **검증 로직 추가**: 예시 키워드 검출 및 경고
4. ✅ **실제 문서 강조**: 참고 문서에서 발견한 내용만 사용

## 💡 해결된 문제

| 문제 | 해결 |
|------|------|
| ❌ 예시 내용(태양광, 디젤) 복사 | ✅ 실제 PDF 문서 분석 |
| ❌ 엉뚱한 내용 출력 | ✅ 문서에 있는 내용만 분석 |
| ❌ 형식 예시와 실제 분석 혼동 | ✅ 명확한 구분 |

## 📊 기능

- **Multi-Agent 분석**: PMC + 섹터 전문가 2단계 검토
- **12개 섹터 지원**: 교육, 보건, 거버넌스, 농촌개발, 물, 에너지, 교통, 도시, 과학기술, 기후행동, 성평등, 인권
- **RAG 기반**: 문서 내 실제 내용 검색 및 인용
- **검증 로직**: 플레이스홀더, 논리적 모순, 예시 복사 자동 검출

## 📋 시스템 요구사항

- GPU (Colab 무료 T4 가능)
- VRAM 8GB 이상
- Python 3.8+

## 📖 사용 예시

1. PDF 업로드
2. 자동 분석 (5-10분)
3. 결과 확인:
   - 0️⃣ 문서 정보
   - 1️⃣ 사업 핵심 정보
   - 2️⃣ Multi-Agent 분석
   - 3️⃣ 통합 권고안
4. TXT/HTML 다운로드

## 🔄 버전 히스토리

- **v3.1** (2025-01-XX): 예시 복사 방지 강화
- **v3.0**: 질문 생성 자동화
- **v2.9**: Multi-Agent 시스템 도입

## 📝 라이선스

MIT License

## 👥 기여

이슈 및 PR 환영합니다!

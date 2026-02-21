## WhyLab 워크스페이스 규칙

### 1. GPU 우선순위 정책
- Python 스크립트 실행 시 **항상 `conda run -n whylab` 환경을 우선** 사용합니다.
- `whylab` conda 환경: Python 3.12 + PyTorch 2.5.1+cu121 (RTX 4070 SUPER CUDA 지원)
- 벤치마크, 학습, 딥러닝 CATE 추정 등 **연산 집약적 작업은 반드시 GPU 환경**에서 실행합니다.
- 일반 Python 명령어: `conda run -n whylab python <script.py>`
- pytest 실행: `conda run -n whylab python -m pytest <test_file.py>`
- GPU 사용 불가 시 자동 CPU 폴백은 허용하되, 사용자에게 알립니다.

### 2. 테스트 실행
- 모든 테스트는 `python -m pytest` 형식으로 실행합니다.
- 새 기능 구현 후 반드시 테스트를 작성하고 통과를 확인합니다.

### 3. 인코딩
- Windows 환경에서 `python -X utf8` 플래그를 사용하여 cp949 에러를 방지합니다.
- 소스 코드 내 이모지 사용 시 print 출력에서 cp949 에러 발생 가능 — `PYTHONIOENCODING=utf-8` 또는 `-X utf8` 필수.

### 4. 보안
- API 키, 토큰 등 민감 정보는 `.env` 파일로 관리하고 `.gitignore`에 등록합니다.
- git push 전 민감 정보 노출 여부를 반드시 확인합니다.

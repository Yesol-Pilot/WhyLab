# Python 3.12 Slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (OpenMP for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사 (engine, whylab, api 패키지 포함)
COPY api ./api
COPY engine ./engine
COPY whylab ./whylab

# PYTHONPATH 설정 (현재 디렉토리를 모듈 경로에 추가)
ENV PYTHONPATH=/app

# 실행 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

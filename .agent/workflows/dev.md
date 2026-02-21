---
description: WhyLab 개발 환경 셋업 및 실행
---

// turbo-all

# WhyLab 개발 워크플로우

## 1단계: 스킬 설치 (새 IDE에서 1회)
```powershell
npx -y antigravity-awesome-skills --path "D:\00.test\PAPER\WhyLab\.agent\skills"
```

## 2단계: Python 환경 (1회)
```powershell
winget install Anaconda.Miniconda3
```

```powershell
conda create -n whylab python=3.11 -y
```

```powershell
conda activate whylab
```

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```powershell
pip install econml doubleml shap scikit-learn pandas numpy matplotlib seaborn duckdb jupyterlab pytest
```

## 3단계: Node.js 대시보드 (1회)
```powershell
cd D:\00.test\PAPER\WhyLab\dashboard && npm install
```

## 일상 실행

### 🚀 서버 원클릭 시작 (가장 많이 사용)
포트 충돌 방지를 위해 기존 프로세스를 먼저 정리한 후, 백엔드와 프론트엔드를 순서대로 시작합니다.

// turbo
1. 포트 정리 (4000, 4001)
```powershell
Get-NetTCPConnection -LocalPort 4000,4001 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
```

// turbo
2. 백엔드 서버 시작 (포트 4001)
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 4001 --reload
```

// turbo
3. 프론트엔드 서버 시작 (포트 4000)
```powershell
cd D:\00.test\PAPER\WhyLab\dashboard && npx next dev -p 4000
```

### 파이프라인 실행
```powershell
python D:\00.test\PAPER\WhyLab\engine\run_pipeline.py
```

### Jupyter 논문 작업
```powershell
cd D:\00.test\PAPER\WhyLab\paper && jupyter lab
```

### 대시보드 개발
```powershell
cd D:\00.test\PAPER\WhyLab\dashboard && npm run dev
```

### 테스트
```powershell
python -m pytest D:\00.test\PAPER\WhyLab\tests\ -v --tb=short
```

### Git 커밋 + 푸시
```powershell
cd D:\00.test\PAPER\WhyLab && git add -A && git commit -m "update" && git push
```

---

## 대시보드 라우트 맵 (확정)

> **출처**: `dashboard/src/components/Sidebar.tsx` menuItems + `dashboard/src/app/**/page.tsx` 전수 조사
> **기준 포트**: http://localhost:4000
> **참고**: `next.config.ts`에서 `basePath: "/WhyLab"`은 `GITHUB_PAGES=1` 빌드 시에만 적용됨. 로컬 `npm run dev`에서는 접두사 없음.

### 핵심 접속 주소

| 메뉴 | URL | 설명 |
|---|---|---|
| **랜딩 페이지** | http://localhost:4000 | 홈 (Hero 섹션) |
| **Overview** | http://localhost:4000/dashboard | 메인 대시보드 개요 |
| **Upload Data** | http://localhost:4000/dashboard/upload | 데이터 업로드 |
| **Discovery** | http://localhost:4000/dashboard/causal-graph | 인과 그래프 탐색 |
| **Simulation** | http://localhost:4000/dashboard/simulator | 시뮬레이터 |
| **Fairness Audit** | http://localhost:4000/dashboard/fairness | 공정성 감사 |
| **Dose-Response** | http://localhost:4000/dashboard/dose-response | 용량-반응 분석 |
| **Policy Simulator** | http://localhost:4000/dashboard/policy-simulator | 정책 시뮬레이터 |
| **Control Room** | http://localhost:4000/dashboard/system | 🎯 관제 센터 (메인) |
| **Knowledge Graph** | http://localhost:4000/dashboard/system/knowledge | 지식 그래프 |
| **Agent Evolution** | http://localhost:4000/dashboard/system/evolution | 에이전트 진화 |
| **Research Cycles** | http://localhost:4000/dashboard/system/cycles | 연구 사이클 이력 |
| **Research Report** | http://localhost:4000/dashboard/system/report | 연구 보고서 |
| **Academic Forum** | http://localhost:4000/dashboard/system/forum | 학술 토론 |
| **Autopilot** | http://localhost:4000/dashboard/system/autopilot | 🚀 자율 연구 실행 |
| **System Health** | http://localhost:4000/dashboard/system/control | 시스템 상태 |
| **Settings** | http://localhost:4000/dashboard/settings | 설정 |

### 추가 페이지 (Sidebar 미등록)

| URL | 설명 |
|---|---|
| http://localhost:4000/live | 실시간 모니터링 |
| http://localhost:4000/system | 시스템 (별도 진입점) |

### 서버 포트 정리

| 서비스 | 포트 | 실행 명령 |
|---|---|---|
| FastAPI 백엔드 | 4001 | `python -m uvicorn api.main:app --port 4001` |
| Next.js 대시보드 | 4000 | `cd dashboard && npm run dev` |

---
description: WhyLab 개발 환경 셋업 및 실행
---

# WhyLab 개발 워크플로우

## 1단계: 스킬 설치 (새 IDE에서 1회)
```powershell
npx -y antigravity-awesome-skills --path "D:\00.test\PAPER\WhyLab\.agent\skills"
```

## 2단계: Python 환경 (1회)
```powershell
winget install Anaconda.Miniconda3
```

// turbo
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
// turbo
```powershell
cd D:\00.test\PAPER\WhyLab\dashboard && npm install
```

## 일상 실행

### 파이프라인 실행
```powershell
conda activate whylab && python D:\00.test\PAPER\WhyLab\engine\run_pipeline.py
```

### Jupyter 논문 작업
```powershell
conda activate whylab && cd D:\00.test\PAPER\WhyLab\paper && jupyter lab
```

### 대시보드 개발
// turbo
```powershell
cd D:\00.test\PAPER\WhyLab\dashboard && npm run dev
```

### 테스트
// turbo
```powershell
conda activate whylab && cd D:\00.test\PAPER\WhyLab\engine && python -m pytest tests/ -v
```

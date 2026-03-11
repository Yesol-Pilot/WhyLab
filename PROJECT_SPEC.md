# 🔬 WhyLab — PROJECT SPEC

> **최종 업데이트**: 2026-03-07 | **유형**: 연구 대시보드
> **상위 문서**: [PAPER PROJECT_SPEC](../PROJECT_SPEC.md) | [마스터 바이블](file:///d:/00.test/FOLDER_BIBLE.md)

---

## 개요

| 항목 | 값 |
|------|------|
| **GitHub** | Yesol-Pilot/WhyLab (🌐 public) |
| **도메인** | whylab.neogenesis.app |
| **브랜치** | main |
| **커밋** | 79개 |

---

## 설명

연구 데이터 시각화 대시보드. 실험 결과를 인터랙티브하게 탐색.

---

## 기술 스택

| 패키지 | 버전 | 용도 |
|--------|------|------|
| next | 16.1.6 | SSR/SSG |
| react / react-dom | 19.2.3 | UI |
| recharts | ^3.7.0 | 차트 |
| reactflow | ^11.11.4 | 노드 그래프 시각화 |
| framer-motion | ^12.34.0 | 애니메이션 |
| lucide-react | ^0.563.0 | 아이콘 |
| katex / @types/katex | ^0.16.28 | 수학 수식 렌더링 |
| clsx + tailwind-merge | — | CSS 유틸 |

---

## 구조

```
WhyLab/
├── dashboard/          ← Next.js 대시보드 앱
│   ├── src/app/        ← App Router 페이지
│   ├── src/components/ ← React 컴포넌트
│   └── package.json
└── .agent/skills/      ← antigravity-awesome-skills (submodule)
```

---

## 배포

Git push → Vercel 자동 배포 (Root Directory: `dashboard/`)

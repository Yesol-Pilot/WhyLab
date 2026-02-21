"""
Gemini LLM Client — WhyLab 연구 AI 래퍼
========================================
Gemini 2.0 Flash API를 활용하여 에이전트의 지능을 강화합니다.

[역할]
- Theorist: KG 갭 분석 → 인과 가설 생성
- 종합 분석: N사이클 누적 결과 리뷰 (5사이클마다)

[비용] 사이클당 ~₩2.3 (사실상 무료)
[Fallback] API 실패 시 기존 템플릿 기반으로 자동 전환
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("whylab.gemini")

# .env 로딩 (python-dotenv 미설치 시 수동 파싱)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def _call_gemini(prompt: str, max_tokens: int = 1024) -> Optional[str]:
    """
    Gemini API를 호출합니다.
    
    Returns:
        생성된 텍스트, 실패 시 None
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY가 설정되지 않았습니다. Fallback 모드.")
        return None
    
    import urllib.request
    import urllib.error
    
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        },
    }).encode("utf-8")
    
    req = urllib.request.Request(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return None
    except urllib.error.HTTPError as e:
        logger.error("Gemini API HTTP 오류: %s %s", e.code, e.reason)
        return None
    except Exception as e:
        logger.error("Gemini API 호출 실패: %s", str(e))
        return None


def generate_hypothesis(kg_context: dict) -> Optional[dict]:
    """
    KG 컨텍스트를 기반으로 Gemini에 가설을 생성 요청합니다.
    
    Args:
        kg_context: {
            "nodes": [{"name": ..., "category": ...}, ...],
            "edges": [{"source": ..., "target": ..., "relation": ..., "weight": ...}, ...],
            "recent_results": [{"ate": ..., "conclusion": ...}, ...],
            "gaps": [{"source": ..., "target": ..., "description": ...}, ...],
        }
    
    Returns:
        {"text": 가설 문장, "reasoning": 추론 근거, "variables": {...}}
        실패 시 None
    """
    # KG 요약을 프롬프트로 구성
    nodes_summary = ", ".join(
        f"{n['name']}({n.get('category', '?')})" 
        for n in kg_context.get("nodes", [])[:15]
    )
    
    edges_summary = "\n".join(
        f"  - {e['source']} →({e.get('relation', '?')}, 신뢰도:{e.get('weight', 0):.1%})→ {e['target']}"
        for e in kg_context.get("edges", [])[:10]
    )
    
    gaps_summary = "\n".join(
        f"  - {g['description']}"
        for g in kg_context.get("gaps", [])[:5]
    )
    
    recent = kg_context.get("recent_results", [])
    recent_summary = "\n".join(
        f"  - ATE={r.get('ate', '?')}, 결론={r.get('conclusion', '?')}"
        for r in recent[:3]
    )
    
    prompt = f"""당신은 인과추론(Causal Inference) 전문 연구자입니다.
아래 Knowledge Graph 현황을 분석하고, 새로운 인과 가설을 1개 생성하세요.

## Knowledge Graph 현황
- 노드: {nodes_summary}
- 관계:
{edges_summary}

## 연구 갭 (미탐색 영역)
{gaps_summary}

## 최근 실험 결과
{recent_summary if recent_summary else "  - 아직 실험 결과 없음"}

## 요청
1. 위 KG의 갭을 기반으로 검증 가능한 인과 가설 1개를 생성하세요.
2. 가설은 "X가 Y에 미치는 인과 효과는 Z 조건에서..."와 같은 구체적인 형태로.
3. **반드시** 아래 JSON 형식으로만 응답하세요:

```json
{{
  "hypothesis": "가설 문장",
  "reasoning": "이 가설을 세운 이유 (1~2문장)",
  "treatment": "처치 변수명",
  "outcome": "결과 변수명",
  "moderator": "조절 변수명 (있으면)"
}}
```"""
    
    response = _call_gemini(prompt, max_tokens=512)
    if not response:
        return None
    
    # JSON 파싱 시도
    try:
        # 코드 블록 안의 JSON 추출
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        
        parsed = json.loads(json_str)
        return {
            "text": parsed.get("hypothesis", response),
            "reasoning": parsed.get("reasoning", ""),
            "treatment": parsed.get("treatment", ""),
            "outcome": parsed.get("outcome", ""),
            "moderator": parsed.get("moderator", ""),
            "source": "gemini",
        }
    except (json.JSONDecodeError, IndexError):
        # JSON 파싱 실패 시 텍스트 그대로 반환
        return {
            "text": response.strip()[:300],
            "reasoning": "LLM 응답을 구조화하지 못함",
            "treatment": "",
            "outcome": "",
            "moderator": "",
            "source": "gemini_raw",
        }


def summarize_cycles(cycle_results: list[dict]) -> Optional[str]:
    """
    N사이클 누적 결과를 Gemini에 종합 분석 요청합니다.
    
    Args:
        cycle_results: 최근 N사이클의 실험 결과 목록
    
    Returns:
        종합 분석 텍스트 (실패 시 None)
    """
    summary_lines = []
    for i, r in enumerate(cycle_results[:10], 1):
        summary_lines.append(
            f"사이클 {i}: ATE={r.get('ate', '?')}, "
            f"방법={r.get('method', '?')}, "
            f"결론={r.get('conclusion', '?')}, "
            f"판정={r.get('verdict', '?')}"
        )
    
    prompt = f"""당신은 인과추론 연구 리뷰어입니다.
아래 연구 사이클 결과를 종합 분석하고 인사이트를 도출하세요.

## 사이클 결과
{chr(10).join(summary_lines)}

## 요청
1. 전체 연구 트렌드를 요약하세요.
2. 가장 신뢰할 수 있는 발견은 무엇인가요?
3. 다음 연구 방향을 제안하세요.
4. 한국어로 300자 이내로 답하세요."""
    
    return _call_gemini(prompt, max_tokens=512)


def evaluate_experiment(result: dict) -> Optional[dict]:
    """
    실험 결과를 Gemini가 비판적으로 평가합니다 (Critic용).
    
    Args:
        result: 실험 결과 딕셔너리
    
    Returns:
        {"critique": "평가 내용", "score": 1~10점}
    """
    prompt = f"""당신은 까다로운 동료 리뷰어(Peer Reviewer)입니다.
아래 실험 결과를 비판적으로 분석하고 피드백을 주세요.

## 실험 개요
- 가설: {result.get('hypothesis_id', '?')}
- 방법론: {result.get('method', '?')} (Estimator: {result.get('estimator', '?')})
- 표본 크기: {result.get('sample_size', 0)}

## 주요 결과
- ATE: {result.get('ate', 0):.2f} (95% CI: {result.get('ate_ci', [])})
- 결론: {result.get('conclusion', '?')}
- 서브그룹 분석:
{json.dumps(result.get('subgroup_analysis', {}), indent=2, ensure_ascii=False)}

## 요청
1. 통계적 유의성과 효과 크기(Effect Size)를 해석하세요.
2. 서브그룹 이질성(Heterogeneity)이 신뢰할만한지 평가하세요.
3. 연구의 한계점이나 추가 검증이 필요한 부분을 지적하세요.
4. JSON 형식으로 응답하세요:
```json
{{
  "critique": "3~4문장의 핵심 비평",
  "score": 7 (1~10점, 10점이 완벽함)
}}
```"""
    
    response = _call_gemini(prompt, max_tokens=512)
    if not response:
        return None
        
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except:
        return {"critique": response[:300], "score": 5}


def is_available() -> bool:
    """Gemini API가 사용 가능한지 확인합니다."""
    return bool(GEMINI_API_KEY)


# ──────────────────────────────────────────────
# 학술 토론 (Forum) 전용 함수들
# ──────────────────────────────────────────────

# 에이전트 페르소나 정의
AGENT_PERSONAS = {
    "Theorist": {
        "name": "Albert",
        "style": "이론적 프레임워크와 인과 모형을 중시하며, 기존 문헌과의 연결성을 강조합니다.",
        "bias": "이론적 일관성",
    },
    "Engineer": {
        "name": "Tesla",
        "style": "실증적 데이터와 통계적 검정력을 중시하며, 방법론적 엄밀성을 강조합니다.",
        "bias": "방법론적 엄밀성",
    },
    "Critic": {
        "name": "Kant",
        "style": "논리적 일관성과 외부 타당도를 중시하며, 재현 가능성을 강조합니다.",
        "bias": "재현 가능성",
    },
}


def generate_debate_topic(kg_context: dict) -> Optional[dict]:
    """
    KG 현재 상태 기반으로 토론 주제를 Gemini가 생성합니다.
    
    Returns:
        {"topic": "주제 문장", "context": "배경 설명", "domain": "연구 분야"}
    """
    nodes_summary = ", ".join(
        f"{n['name']}({n.get('category', '?')})"
        for n in kg_context.get("nodes", [])[:10]
    )
    edges_summary = "\n".join(
        f"  - {e['source']} →({e.get('relation', '?')})→ {e['target']} (신뢰도:{e.get('weight', 0):.0%})"
        for e in kg_context.get("edges", [])[:8]
    )
    recent = kg_context.get("recent_results", [])
    recent_summary = "\n".join(
        f"  - {r.get('hypothesis', '?')}: ATE={r.get('ate', '?')}, 판정={r.get('verdict', '?')}"
        for r in recent[:3]
    ) or "  - 아직 없음"

    prompt = f"""당신은 인과추론(Causal Inference) 학술 포럼의 사회자입니다.
아래 Knowledge Graph 현황과 최근 연구 결과를 검토하고,
연구자들이 토론할 학술 논제를 1개 생성하세요.

## Knowledge Graph 현황
- 노드: {nodes_summary}
- 관계:
{edges_summary}

## 최근 연구 결과
{recent_summary}

## 요청
- 논제는 해당 연구 결과의 타당성, 일반화 가능성, 또는 후속 연구 방향에 대한 것이어야 합니다.
- 반드시 아래 JSON 형식으로만 응답하세요:
```json
{{
  "topic": "토론 논제 (한 문장)",
  "context": "배경 설명 (1~2문장)",
  "domain": "관련 연구 분야"
}}
```"""

    response = _call_gemini(prompt, max_tokens=512)
    if not response:
        return None

    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"topic": response.strip()[:200], "context": "", "domain": "인과추론"}


def generate_debate_response(
    role: str,
    topic: str,
    phase: str,
    previous_statements: list[dict],
    kg_context: dict,
) -> Optional[str]:
    """
    학술 토론에서 에이전트의 발언을 Gemini가 생성합니다.
    
    Args:
        role: "Theorist" | "Engineer" | "Critic"
        topic: 토론 논제
        phase: "opening" | "rebuttal" | "closing"
        previous_statements: [{"role": ..., "content": ...}, ...]
        kg_context: KG 현황 딕셔너리
    
    Returns:
        생성된 발언 텍스트 (실패 시 None)
    """
    persona = AGENT_PERSONAS.get(role, AGENT_PERSONAS["Theorist"])
    
    prev_text = "\n".join(
        f"[{s['role']}({AGENT_PERSONAS.get(s['role'], {}).get('name', '?')})]: {s['content']}"
        for s in previous_statements
    ) or "  (첫 발언입니다)"

    phase_instruction = {
        "opening": "논제에 대한 자신의 입장을 명확히 밝히고, 근거를 제시하세요.",
        "rebuttal": "이전 발언자들의 주장에 대해 동의/반박하며, 추가 논거를 제시하세요.",
        "closing": "토론 내용을 종합하여 최종 입장을 정리하고, 후속 연구 방향을 제안하세요.",
    }

    nodes_summary = ", ".join(
        f"{n['name']}" for n in kg_context.get("nodes", [])[:8]
    )

    prompt = f"""당신은 인과추론 연구자 '{persona['name']}'({role})입니다.
성향: {persona['style']}
주요 관심사: {persona['bias']}

## 토론 논제
{topic}

## KG의 주요 변수
{nodes_summary}

## 이전 발언
{prev_text}

## 현재 단계: {phase}
{phase_instruction.get(phase, phase_instruction['opening'])}

## 규칙
- 3~5문장으로 답하세요.
- 학술적이지만 명료하게 써 주세요.
- 구체적 변수명이나 분석 방법을 언급하세요.
- 반드시 순수 텍스트로만 응답하세요 (JSON/마크다운 불필요)."""

    return _call_gemini(prompt, max_tokens=512)


def generate_consensus(topic: str, statements: list[dict]) -> Optional[dict]:
    """
    토론 내용을 종합하여 합의를 Gemini가 도출합니다.
    
    Returns:
        {"label": "합의|조건부 합의|이견", "summary": "합의 내용", "next_steps": ["후속 연구 1", ...]}
    """
    debate_text = "\n".join(
        f"[{s['role']}({s.get('phase', '?')})]: {s['content']}"
        for s in statements
    )

    prompt = f"""당신은 학술 토론의 사회자입니다.
아래 토론 내용을 종합하여 합의를 도출하세요.

## 논제
{topic}

## 토론 내용
{debate_text}

## 요청
반드시 아래 JSON 형식으로만 응답하세요:
```json
{{
  "label": "합의" 또는 "조건부 합의" 또는 "이견",  
  "summary": "합의/불합의 내용 요약 (2~3문장)",
  "next_steps": ["후속 연구 제안 1", "후속 연구 제안 2"]
}}
```"""

    response = _call_gemini(prompt, max_tokens=512)
    if not response:
        return None

    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"label": "이견", "summary": response.strip()[:200], "next_steps": []}


def generate_evolution_strategy(
    role: str, performance: dict, kg_context: dict
) -> Optional[dict]:
    """
    에이전트 성과와 KG 컨텍스트 기반으로 다음 세대 전략을 Gemini가 제안합니다.
    
    Returns:
        {"specialization": "전문화 방향", "focus_area": "집중 분야", "reasoning": "이유"}
    """
    scores_text = "\n".join(
        f"  - {k}: {v.get('score', 0):.1f}점 (가중치 {v.get('weight', 0):.0%})"
        for k, v in performance.get("scores", {}).items()
    )

    nodes_summary = ", ".join(
        f"{n['name']}({n.get('category', '?')})"
        for n in kg_context.get("nodes", [])[:10]
    )

    prompt = f"""당신은 AI 연구 에이전트 진화 시스템의 설계자입니다.
아래 에이전트의 성과를 분석하고, 다음 세대의 전문화 방향을 제안하세요.

## 에이전트 정보
- 역할: {role}
- 총점: {performance.get('total_score', 0):.1f}점
- 세부 평가:
{scores_text}

## Knowledge Graph 상태
- 주요 변수: {nodes_summary}

## 요청
- 이 에이전트가 다음 세대에서 어떤 방향으로 전문화되어야 하는지 제안하세요.
- 반드시 아래 JSON 형식으로만 응답하세요:
```json
{{
  "specialization": "전문화 방향 (예: 비선형 인과 분석, 도구변수 전문가 등)",
  "focus_area": "집중 분야 (예: 이질적 처리효과, 민감도 분석 등)",
  "reasoning": "이 전문화를 제안하는 이유 (1~2문장)"
}}
```"""

    response = _call_gemini(prompt, max_tokens=512)
    if not response:
        return None

    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"specialization": "범용", "focus_area": "일반", "reasoning": response.strip()[:200]}

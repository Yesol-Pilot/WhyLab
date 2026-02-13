"""WhyLab Membrane Server — MCP Protocol Interface.

이 서버는 WhyLab 엔진을 외부 에이전트(Claude Desktop 등)와 연결하는
표준 인터페이스(Membrane) 역할을 합니다.
"""

from mcp.server.fastmcp import FastMCP
import json
from pathlib import Path
from typing import Any, Dict

# WhyLab 엔진 모듈 임포트
from engine.pipeline import run_pipeline
from engine.config import WhyLabConfig

# MCP 서버 초기화 (서버 이름: WhyLab)
mcp = FastMCP("WhyLab")

# 전역 설정 로드
config = WhyLabConfig()

@mcp.resource("whylab://data/latest")
def get_latest_data() -> str:
    """최신 분석 결과 JSON 데이터를 반환합니다."""
    json_path = config.paths.dashboard_data_dir / "latest.json"
    if not json_path.exists():
        return json.dumps({"error": "No data found. Run pipeline first."}, ensure_ascii=False)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return f.read()

@mcp.tool()
def run_analysis(scenario: str = "A") -> str:
    """특정 시나리오에 대한 인과추론 파이프라인을 실행합니다.

    Args:
        scenario: "A" (Credit Limit) 또는 "B" (Marketing Coupon).
    """
    try:
        result = run_pipeline(scenario=scenario)
        # 결과 요약 반환 (전체 데이터는 너무 큼)
        summary = {
            "ate": result.get("ate"),
            "model_type": result.get("model_type"),
            "sensitivity": result.get("sensitivity_results", {}).get("status"),
            "json_path": result.get("json_path")
        }
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error running analysis: {str(e)}"

@mcp.tool()
def simulate_intervention(intervention_value: float) -> str:
    """(실험적) 현재 모델에서 특정 개입 값에 대한 결과를 시뮬레이션합니다.
    
    Args:
        intervention_value: 개입의 강도 (예: 신용한도 증가량).
    """
    # TODO: WhatIfSimulator 로직을 Python 엔진으로 이식해야 함.
    # 현재는 Mock 반환
    return f"Simulation with intervention={intervention_value} completed. (Mock)"

if __name__ == "__main__":
    # stdio 모드로 서버 실행
    mcp.run()

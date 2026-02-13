# -*- coding: utf-8 -*-
"""Discovery Agent — 인과 구조 발견을 위한 Nucleus Module.

데이터와 메타데이터를 분석하여 변수 간의 인과 관계(DAG)를 스스로 수립합니다.
LLM의 상식적 추론(Prior Knowledge)과 통계적 알고리즘(PC Algorithm)을 결합하는
하이브리드 발견 전략을 사용합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import networkx as nx

from engine.config import WhyLabConfig

class DiscoveryAgent:
    """인과 구조(DAG)를 자율적으로 발견하는 에이전트 (Nucleus)."""

    def __init__(self, config: WhyLabConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("whylab.agents.discovery")
        self._llm_client = None  # 추후 LLM 클라이언트 연동 (MCP 등)

    def discover(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> nx.DiGraph:
        """데이터로부터 인과 그래프를 발견합니다.

        Args:
            df: 분석 대상 데이터프레임.
            metadata: 컬럼 설명 등 메타데이터.

        Returns:
            NetworkX DiGraph 객체 (발견된 DAG).
        """
        self.logger.info("🧠 Nucleus(Discovery) 활성화: 데이터 분석 시작 (Rows: %d)", len(df))

        # 1. LLM 기반 사전 지식(Prior Knowledge) 수립
        prior_dag = self._reason_with_llm(metadata)
        
        # 2. 통계적 인과 발견 (PC Algorithm)
        # 실제 causal-learn 라이브러리가 있다면 사용, 없다면 Heuristic fallback
        stat_dag = self._discover_statistically(df)

        # 3. 하이브리드 병합 (Ensemble)
        final_dag = self._merge_graphs(prior_dag, stat_dag)
        
        self.logger.info("✨ 인과 구조 발견 완료 (Nodes: %d, Edges: %d)",
                         final_dag.number_of_nodes(), final_dag.number_of_edges())
        return final_dag

    def _reason_with_llm(self, metadata: Dict[str, Any]) -> nx.DiGraph:
        """LLM을 사용하여 변수 간의 상식적인 인과관계를 추론합니다."""
        self.logger.info("   [1] LLM Reasoning: 변수 의미론적 분석 중...")
        
        # TODO: 실제 LLM API 호출 (OpenAI / Gemini)
        # 현재는 메타데이터 기반의 규칙(Rule-based) 모의 추론으로 대체
        
        dag = nx.DiGraph()
        nodes = metadata.get("feature_names", []) + [
            metadata.get("treatment_col"), metadata.get("outcome_col")
        ]
        
        # 노드 추가
        for node in nodes:
            if node:
                dag.add_node(node)
        
        # Mock Logic: "나이(age)는 다른 변수의 원인이 될 수 있지만, 결과가 될 순 없다."
        if "age" in nodes:
            if "credit_limit" in nodes:
                dag.add_edge("age", "credit_limit")
            if "is_default" in nodes:
                dag.add_edge("age", "is_default")
                
        self.logger.info("       LLM 가설 수립 완료.")
        return dag

    def _discover_statistically(self, df: pd.DataFrame) -> nx.DiGraph:
        """PC 알고리즘 등 통계적 방법으로 인과관계를 발견합니다."""
        self.logger.info("   [2] Statistical Discovery: 조건부 독립성 검정 중...")
        
        dag = nx.DiGraph()
        columns = df.columns.tolist()
        dag.add_nodes_from(columns)
        
        # Placeholder: 상관계수 기반으로 엣지 후보 생성 (실제 PC 알고리즘 대체)
        # 실제 구현에선 causal-learn의 cdt.causality.graph.PC 등을 사용해야 함.
        corr_matrix = df.corr().abs()
        threshold = 0.3  # 임계값
        
        for i, col_a in enumerate(columns):
            for j, col_b in enumerate(columns):
                if i >= j: continue
                
                # 강한 상관관계가 있으면 엣지 연결 (방향은 아직 미정)
                if corr_matrix.iloc[i, j] > threshold:
                    # 간단한 시간적/논리적 선후관계 가정 (Heuristic)
                    # 예: 나이 -> 소득
                    if col_a == "age":
                        dag.add_edge(col_a, col_b)
                    elif col_b == "age":
                        dag.add_edge(col_b, col_a)
                    else:
                        # 방향을 모를 때는 일단 양방향(또는 무방향)이 원칙이나
                        # 여기선 임의로 i -> j (추후 LLM이 교정)
                        dag.add_edge(col_a, col_b)
                        
        self.logger.info("       통계적 패턴 추출 완료.")
        return dag

    def _merge_graphs(self, prior: nx.DiGraph, stat: nx.DiGraph) -> nx.DiGraph:
        """LLM의 가설(Prior)과 통계적 발견(Data)을 통합합니다."""
        self.logger.info("   [3] Hybrid Fusion: 가설과 데이터의 통합")
        
        # 기본 전략: 통계적 발견을 존중하되, LLM의 상식으로 방향을 교정
        merged = stat.copy()
        
        # LLM의 강력한 제약조건(Hard Constraints) 적용
        # 예: Prior에 있는 엣지는 반드시 포함하거나 방향을 강제
        for u, v in prior.edges():
            if not merged.has_edge(u, v):
                # 데이터에선 약했지만 상식적으로 확실하면 추가
                if not merged.has_edge(v, u): # 역방향이 없다면
                    merged.add_edge(u, v)
            elif merged.has_edge(v, u):
                # 데이터가 역방향을 가리키면, 상식(LLM)을 우선하여 뒤집음
                merged.remove_edge(v, u)
                merged.add_edge(u, v)
                
        return merged

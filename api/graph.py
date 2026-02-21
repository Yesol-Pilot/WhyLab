"""
Knowledge Graph v2 — JSON 영속화 지원
========================================
NetworkX 기반 in-memory KG + JSON 파일 자동 저장/복원.

[변경사항]
- save() / load(): JSON 파일 기반 영속화
- add_verified_edge(): 실험 결과 엣지 추가 + 자동 저장
- 서버 시작 시 기존 KG 복원
"""
import json
import os
import logging
import time
import threading
from datetime import datetime, timezone, timedelta

import networkx as nx

logger = logging.getLogger("whylab.kg")

# 한국 표준시 (KST = UTC+9)
KST = timezone(timedelta(hours=9))

# KG 저장 경로
KG_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KG_STATE_FILE = os.path.join(KG_DATA_DIR, "kg_state.json")


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.initialized = False
        self._save_lock = threading.Lock()
        self._dirty = False
        
        # 시작 시 기존 상태 복원 시도
        self._load_or_seed()
    
    def _load_or_seed(self):
        """저장된 KG 파일이 있으면 복원, 없으면 Seed 데이터 초기화."""
        if os.path.exists(KG_STATE_FILE):
            try:
                self.load()
                logger.info(f"[KG] 기존 상태 복원: {self.graph.number_of_nodes()}노드, {self.graph.number_of_edges()}엣지")
                return
            except Exception as e:
                logger.warning(f"[KG] 상태 복원 실패, Seed로 초기화: {e}")
        
        self.initialize_seed_data()
    
    def initialize_seed_data(self):
        """LaLonde 데이터셋 기반 초기 지식 그래프 구축."""
        if self.initialized:
            return
        
        # 핵심 개념 노드
        concepts = [
            ("Treatment", "Job Training Program"),
            ("Outcome", "Real Earnings (1978)"),
            ("Confounder", "Age"),
            ("Confounder", "Education"),
            ("Confounder", "Race"),
            ("Confounder", "Marital Status"),
        ]
        for category, name in concepts:
            self.graph.add_node(name, type="Concept", category=category)
        
        # 초기 인과관계 (검증 대상)
        seed_edges = [
            ("Job Training Program", "Real Earnings (1978)", "increases", 0.5),
            ("Age", "Real Earnings (1978)", "affects", 0.3),
            ("Education", "Real Earnings (1978)", "increases", 0.4),
            ("Race", "Real Earnings (1978)", "correlates", 0.2),
        ]
        for src, dst, rel, w in seed_edges:
            self.graph.add_edge(src, dst, relation=rel, weight=w, verified=False)
        
        self.initialized = True
        self.save()
        return f"KG 초기화: {self.graph.number_of_nodes()}노드, {self.graph.number_of_edges()}엣지"
    
    def add_verified_edge(self, source: str, target: str, **attrs):
        """
        검증된 실험 결과를 엣지로 추가하고 자동 저장.
        
        Args:
            source: 원인 노드
            target: 결과 노드
            **attrs: 엣지 속성 (ate, p_value, method, experiment_id 등)
        """
        attrs.setdefault("verified", True)
        attrs.setdefault("added_at", datetime.now(KST).isoformat())
        
        # 노드가 없으면 자동 생성
        if source not in self.graph:
            self.graph.add_node(source, type="Concept", category="Discovered")
        if target not in self.graph:
            self.graph.add_node(target, type="Concept", category="Discovered")
        
        self.graph.add_edge(source, target, **attrs)
        self._dirty = True
        self.save()
        
        logger.info(f"[KG] 엣지 추가: {source} → {target} (ATE={attrs.get('ate', 'N/A')})")
    
    def get_stats(self):
        """KG 통계."""
        edges = list(self.graph.edges(data=True))
        verified = sum(1 for _, _, d in edges if d.get("verified"))
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "verified_edges": verified,
            "density": round(nx.density(self.graph), 4),
            "total_hypotheses": sum(1 for _, _, d in edges if d.get("hypothesis_id")),
            "total_experiments": sum(1 for _, _, d in edges if d.get("experiment_id")),
        }
    
    def get_graph_data(self):
        """프론트엔드 시각화용 데이터 포맷."""
        nodes = [
            {"id": n, "label": n, "group": self.graph.nodes[n].get("category", "Unknown")}
            for n in self.graph.nodes
        ]
        links = [
            {
                "source": u, "target": v,
                "label": d.get("relation", ""),
                "weight": d.get("weight", 0),
                "verified": d.get("verified", False),
                "ate": d.get("ate"),
            }
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "links": links}
    
    # ─── 영속화 ───
    
    def save(self):
        """KG를 JSON 파일에 저장."""
        with self._save_lock:
            os.makedirs(KG_DATA_DIR, exist_ok=True)
            
            data = {
                "saved_at": datetime.now(KST).isoformat(),
                "nodes": [
                    {"id": n, **{k: str(v) for k, v in d.items()}}
                    for n, d in self.graph.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, **self._serialize_edge(d)}
                    for u, v, d in self.graph.edges(data=True)
                ],
            }
            
            # 원자적 쓰기 (임시 파일 → 이동)
            tmp_path = KG_STATE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Windows에서는 rename 전 기존 파일 삭제 필요
            if os.path.exists(KG_STATE_FILE):
                os.remove(KG_STATE_FILE)
            os.rename(tmp_path, KG_STATE_FILE)
            
            self._dirty = False
    
    def load(self):
        """JSON 파일에서 KG 복원."""
        with open(KG_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.graph.clear()
        
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            # 숫자형 복원
            for k in ("weight", "ate", "p_value"):
                if k in edge:
                    try:
                        edge[k] = float(edge[k])
                    except (ValueError, TypeError):
                        pass
            if "verified" in edge:
                edge["verified"] = edge["verified"] in (True, "True", "true")
            self.graph.add_edge(src, tgt, **edge)
        
        self.initialized = True
        logger.info(f"[KG] 복원 완료: {self.graph.number_of_nodes()}노드, {self.graph.number_of_edges()}엣지")
    
    @staticmethod
    def _serialize_edge(data: dict) -> dict:
        """엣지 속성을 JSON 직렬화 가능한 형태로 변환."""
        result = {}
        for k, v in data.items():
            if isinstance(v, (int, float, bool, str, type(None))):
                result[k] = v
            elif isinstance(v, (list, dict)):
                result[k] = v
            else:
                result[k] = str(v)
        return result


# 글로벌 싱글턴 인스턴스
kg = KnowledgeGraph()

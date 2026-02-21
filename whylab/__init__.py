# -*- coding: utf-8 -*-
"""WhyLab: AI-Driven Causal Inference Engine.

3줄 코드로 인과 분석을 수행할 수 있는 간편 API를 제공합니다.

사용법:
    import whylab
    result = whylab.analyze("data.csv", treatment="T", outcome="Y")
    result.summary()
"""

from whylab.api import analyze, CausalResult
from engine.cells.dose_response_cell import DoseResponseCell
from engine.cells.fairness_audit_cell import FairnessAuditCell
from engine.cells.deep_cate_cell import DeepCATECell
from engine.agents.mac_discovery import MACDiscoveryAgent

__version__ = "1.0.0"
__all__ = [
    "analyze", "CausalResult",
    "DoseResponseCell", "FairnessAuditCell", "DeepCATECell",
    "MACDiscoveryAgent",
    "__version__"
]

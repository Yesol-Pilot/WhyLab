from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    session_id: str
    treatment: str
    outcome: str
    confounders: List[str]
    variables: Optional[List[str]] = None

class FairnessRequest(AnalysisRequest):
    sensitive_attrs: List[str]

class DiscoveryRequest(BaseModel):
    session_id: str
    variables: Optional[List[str]] = None

class SimulationRequest(BaseModel):
    session_id: str
    intensity: float  # 처치 증가량
    target_percent: float # 대상 유저 비율 (e.g. 20%)
    cost_per_unit: float = 1.0 # 비용 계수

# ──────────────────────────────────────────────
# Response Schemas (ORM-based)
# ──────────────────────────────────────────────

class DatasetBase(BaseModel):
    filename: str
    rows: int
    columns: List[str]

    class Config:
        from_attributes = True

class SessionBase(BaseModel):
    id: str
    created_at: datetime
    datasets: List[DatasetBase] = []

    class Config:
        from_attributes = True

class AnalysisResultBase(BaseModel):
    analysis_type: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class AgentBase(BaseModel):
    id: str
    name: str
    role: str
    status: str
    generation: int = 1
    parent_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class SystemLogBase(BaseModel):
    id: int
    agent_id: Optional[str]
    level: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True

class SessionResponse(BaseModel):
    id: str
    created_at: datetime
    datasets: List[DatasetBase]
    analyses: List[AnalysisResultBase]

    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: List[str]
    preview: List[Dict[str, Any]]

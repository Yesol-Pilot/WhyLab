from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base

class ProjectSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True) # UUID
    created_at = Column(DateTime, default=datetime.utcnow)
    
    datasets = relationship("Dataset", back_populates="session", cascade="all, delete-orphan")
    analyses = relationship("AnalysisResult", back_populates="session", cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    filename = Column(String)
    file_path = Column(String) # 로컬 저장 경로
    rows = Column(Integer)
    columns = Column(JSON) # 컬럼 이름 리스트
    
    session = relationship("ProjectSession", back_populates="datasets")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    analysis_type = Column(String) # 'dose_response', 'discovery', 'fairness'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 분석 설정 (Request Body Snapshot)
    config = Column(JSON)
    
    # 분석 결과 (JSON Serializable)
    result_data = Column(JSON)
    
    # 학습된 모델 파일 경로 (Pickle, Optional)
    model_path = Column(String, nullable=True)
    
    session = relationship("ProjectSession", back_populates="analyses")

class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, index=True) # e.g., "theorist-1"
    name = Column(String) # "Albert"
    role = Column(String) # "Theorist", "Engineer", "Critic"
    status = Column(String, default="IDLE") # "IDLE", "WORKING", "ERROR"
    generation = Column(Integer, default=1) # 진화 세대 (v1, v2, ...)
    parent_id = Column(String, ForeignKey("agents.id"), nullable=True) # 부모 에이전트 (Self-Reference)
    config = Column(JSON) # Capability, Model Info
    created_at = Column(DateTime, default=datetime.utcnow)
    
    parent = relationship("Agent", remote_side=[id], backref="children")
    logs = relationship("SystemLog", back_populates="agent")

class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=True)
    level = Column(String) # "INFO", "WARNING", "ERROR"
    message = Column(String)
    details = Column(JSON, nullable=True) # Stack trace or structured data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    agent = relationship("Agent", back_populates="logs")

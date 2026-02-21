from sqlalchemy.orm import Session
import uuid
from . import models, schemas
import json

def create_session(db: Session):
    session_id = str(uuid.uuid4())
    db_session = models.ProjectSession(id=session_id)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_session(db: Session, session_id: str):
    return db.query(models.ProjectSession).filter(models.ProjectSession.id == session_id).first()

def create_dataset(db: Session, session_id: str, filename: str, file_path: str, rows: int, columns: list):
    db_dataset = models.Dataset(
        session_id=session_id,
        filename=filename,
        file_path=file_path,
        rows=rows,
        columns=columns # JSON serialization handled by SQLAlchemy if dialect supports, else passed as list/dict
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def get_dataset(db: Session, session_id: str):
    return db.query(models.Dataset).filter(models.Dataset.session_id == session_id).first()

def create_analysis_result(db: Session, session_id: str, analysis_type: str, config: dict, result: dict, model_path: str = None):
    # 기존 결과가 있으면 삭제 (덮어쓰기 정책)
    existing = db.query(models.AnalysisResult).filter(
        models.AnalysisResult.session_id == session_id,
        models.AnalysisResult.analysis_type == analysis_type
    ).first()
    
    if existing:
        db.delete(existing)
        db.commit()

    db_analysis = models.AnalysisResult(
        session_id=session_id,
        analysis_type=analysis_type,
        config=config,
        result_data=result,
        model_path=model_path
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_analysis_result(db: Session, session_id: str, analysis_type: str):
    return db.query(models.AnalysisResult).filter(
        models.AnalysisResult.session_id == session_id,
        models.AnalysisResult.analysis_type == analysis_type
    ).first()

# ──────────────────────────────────────────────
# System / Agent CRUD
# ──────────────────────────────────────────────

def create_agent(db: Session, agent_id: str, name: str, role: str, config: dict = None, generation: int = 1, parent_id: str = None):
    agent = models.Agent(id=agent_id, name=name, role=role, config=config, status="IDLE", generation=generation, parent_id=parent_id)
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent

def get_agents(db: Session):
    return db.query(models.Agent).all()

def update_agent_status(db: Session, agent_id: str, status: str):
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if agent:
        agent.status = status
        db.commit()
        db.refresh(agent)
    return agent

def create_log(db: Session, agent_id: str, level: str, message: str, details: dict = None):
    log = models.SystemLog(agent_id=agent_id, level=level, message=message, details=details)
    db.add(log)
    db.commit()
    return log

def get_logs(db: Session, limit: int = 100):
    return db.query(models.SystemLog).order_by(models.SystemLog.created_at.desc()).limit(limit).all()

# -*- coding: utf-8 -*-
"""WhyLab Dashboard Backend API (Persistence Enabled)."""

import sys
import uuid
import logging
import io
import os
import shutil
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api import models, schemas, crud
from api.database import SessionLocal, engine, get_db

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whylab-api")

# DB 테이블 생성 (Production에서는 Migration 도구 사용 권장)
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="WhyLab API", version="2.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 서버 시작 시 자율 연구(Autopilot) 자동 실행 ──
@app.on_event("startup")
def _auto_start_autopilot():
    """서버 부팅 완료 후 Autopilot을 자동으로 시작합니다."""
    import threading
    def _delayed_start():
        import time
        time.sleep(3)  # DB/서버 초기화 대기
        try:
            from api.agents.autopilot import autopilot
            if not autopilot.running:
                result = autopilot.start(db_factory=SessionLocal)
                logger.info(f"🚀 [AUTOPILOT] 자동 시작 완료: {result}")
        except Exception as e:
            logger.error(f"❌ [AUTOPILOT] 자동 시작 실패: {e}")
    threading.Thread(target=_delayed_start, daemon=True).start()

# ... (rest of code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=4001, reload=True)

# 파일 저장 경로 설정
UPLOAD_DIR = ROOT / "data" / "uploads"
MODEL_DIR = ROOT / "data" / "models"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "version": "2.0.0", "persistence": "sqlite"}

@app.get("/session/{session_id}", response_model=schemas.SessionResponse)
def get_session_info(session_id: str, db: Session = Depends(get_db)):
    """세션 상태 및 분석 이력 조회 (복구용)."""
    session = crud.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.post("/upload", response_model=schemas.UploadResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """CSV 파일 업로드, 로컬 저장, DB 메타데이터 기록."""
    try:
        # 1. 세션 생성
        session = crud.create_session(db)
        session_id = session.id
        
        # 2. 파일 저장
        safe_filename = f"{session_id}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. 데이터 로드 및 미리보기 (검증)
        df = pd.read_csv(file_path)
        # 결측치 처리 (데모용)
        df_clean = df.dropna()
        if len(df) != len(df_clean):
            # 덮어쓰기
            df_clean.to_csv(file_path, index=False)
            df = df_clean

        # 4. DB 기록
        crud.create_dataset(
            db=db,
            session_id=session_id,
            filename=file.filename,
            file_path=str(file_path),
            rows=len(df),
            columns=list(df.columns)
        )
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head().to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def get_df_from_db(session_id: str, db: Session) -> pd.DataFrame:
    dataset = crud.get_dataset(db, session_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found for this session")
    
    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=500, detail="Data file lost on server")
        
    return pd.read_csv(dataset.file_path)

@app.post("/analysis/dose-response")
async def analyze_dose_response(req: schemas.AnalysisRequest, db: Session = Depends(get_db)):
    try:
        df = get_df_from_db(req.session_id, db)
        
        from engine.cells.dose_response_cell import DoseResponseCell, DoseResponseConfig
        
        cell = DoseResponseCell(dr_config=DoseResponseConfig(n_grid_points=50))
        input_data = {
            "dataframe": df,
            "treatment_col": req.treatment,
            "outcome_col": req.outcome,
            "feature_names": req.confounders
        }
        
        result = cell.execute(input_data)
        
        # 모델 저장 (Pickle)
        model_filename = f"{req.session_id}_dose_response.joblib"
        model_path = MODEL_DIR / model_filename
        
        # Cell 전체를 저장 (Helper function needed to pickle cell properly?)
        # For simplicity, we save the trained 'response_model' if it exists, or the cell itself.
        # But cell object might be large or unpicklable if it has logger etc.
        # DoseResponseCell handles numpy, it should be fine.
        # *중요*: Logger는 pickle 안될 수 있으므로 처리가 필요하지만, 여기선 joblib이 강력하므로 시도.
        # 안전하게는 response_model과 config만 저장하는 것이 좋음.
        
        # cell.logger 제거 후 저장
        cell.logger = None 
        joblib.dump(cell, model_path)
        
        res = result["dose_response"]
        serializable_res = {
            "t_grid": res["t_grid"],
            "dr_curve": res["dr_curve"],
            "ci_lower": res.get("ci_lower"),
            "ci_upper": res.get("ci_upper"),
            "optimal_dose": float(res["optimal_dose"]),
            "optimal_effect": float(res["optimal_effect"]),
            "has_effect": bool(res["has_effect"])
        }
        
        # DB 저장
        crud.create_analysis_result(
            db=db,
            session_id=req.session_id,
            analysis_type="dose_response",
            config=req.dict(),
            result=serializable_res,
            model_path=str(model_path)
        )
        
        return {"status": "success", "result": serializable_res}
        
    except Exception as e:
        logger.error(f"Dose-response failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/discovery")
async def analyze_discovery(req: schemas.DiscoveryRequest, db: Session = Depends(get_db)):
    try:
        df = get_df_from_db(req.session_id, db)
        vars_to_use = req.variables if req.variables else list(df.columns)
        
        from engine.agents.mac_discovery_agent import MACDiscoveryAgent
        agent = MACDiscoveryAgent()
        
        dag = await agent.discover_causal_structure(df, variable_names=vars_to_use)
        
        edges = [{"source": e.source, "target": e.target} for e in dag.edges]
        nodes = [{"id": v, "label": v} for v in vars_to_use]
        
        res_data = {
             "nodes": nodes,
             "edges": edges,
             "consensus_level": dag.consensus_level,
             "stability_scores": dag.stability_scores
        }
        
        crud.create_analysis_result(
            db=db,
            session_id=req.session_id,
            analysis_type="discovery",
            config=req.dict(),
            result=res_data
        )
        
        return {"status": "success", "result": res_data}
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/fairness")
async def analyze_fairness(req: schemas.FairnessRequest, db: Session = Depends(get_db)):
    try:
        df = get_df_from_db(req.session_id, db)
        
        from engine.cells.meta_learner_cell import TLearner
        from engine.config import WhyLabConfig
        from engine.cells.fairness_audit_cell import FairnessAuditCell
        
        X = df[req.confounders].values
        T = df[req.treatment].values
        Y = df[req.outcome].values
        
        config = WhyLabConfig()
        learner = TLearner(config=config)
        learner.fit(X, T, Y)
        cate = learner.predict_cate(X)
        
        cell = FairnessAuditCell()
        audit_results = cell.audit(cate, df, req.sensitive_attrs)
        
        serialized = []
        for res in audit_results:
            serialized.append({
                "attribute": res.group_name,
                "overall_cate": float(res.overall_cate),
                "is_fair": res.is_fair,
                "metrics": res.metrics,
                "subgroups": [ 
                    {"name": str(k), "mean_cate": float(v.mean_cate), "size": int(v.size)} 
                    for k, v in res.subgroups.items() 
                ]
            })
            
        crud.create_analysis_result(
            db=db,
            session_id=req.session_id,
            analysis_type="fairness",
            config=req.dict(),
            result=serialized
        )
            
        return {"status": "success", "result": serialized}
        
    except Exception as e:
        logger.error(f"Fairness failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/simulate")
async def simulate_policy(req: schemas.SimulationRequest, db: Session = Depends(get_db)):
    try:
        # DB에서 모델 경로 조회
        analysis = crud.get_analysis_result(db, req.session_id, "dose_response")
        if not analysis or not analysis.model_path:
             raise HTTPException(status_code=400, detail="Dose-Response analysis required first.")
             
        # 모델 로드
        if not os.path.exists(analysis.model_path):
             raise HTTPException(status_code=500, detail="Model file lost.")
             
        cell = joblib.load(analysis.model_path)
        # config = analysis.config # DB에 저장된 config 사용 가능
        
        # 데이터 로드
        df = get_df_from_db(req.session_id, db)
        
        # 기존 main.py의 로직 재사용
        # (models.py의 AnalysisResult.config는 JSON이므로 dict로 자동 변환됨)
        config = analysis.config 
        
        n_users = len(df)
        n_target = int(n_users * (req.target_percent / 100))
        if n_target == 0: n_target = 1
            
        df_sorted = df.sort_values(by=config['outcome'])
        target_indices = df_sorted.index[:n_target]
        
        X_target = df.loc[target_indices, config['confounders']].values
        T_target = df.loc[target_indices, config['treatment']].values
        
        T_new = T_target + req.intensity
        
        try:
            Y_pred_old = cell.predict(X_target, T_target)
            Y_pred_new = cell.predict(X_target, T_new)
        except ValueError:
            raise HTTPException(status_code=400, detail="Prediction model not ready.")
            
        benefit = np.sum(Y_pred_new - Y_pred_old)
        cost = req.intensity * req.cost_per_unit * n_target
        net_profit = benefit - cost
        roi = (net_profit / (cost + 1e-10)) * 100
        avg_outcome_change = np.mean(Y_pred_new - Y_pred_old)
        
        # Sensitivity
        sensitivity_data = []
        intensity_range = np.linspace(0, 2000, 11)
        for val in intensity_range:
            T_sens = T_target + val
            try:
                Y_sens = cell.predict(X_target, T_sens)
                sens_benefit = np.sum(Y_sens - Y_pred_old)
                sens_cost = val * req.cost_per_unit * n_target
                sens_profit = sens_benefit - sens_cost
                sens_risk = (val / 2000.0) * 0.1
                sensitivity_data.append({
                    "intensity": float(val),
                    "profit": float(sens_profit),
                    "risk": float(sens_risk * 100)
                })
            except:
                continue

        return {
            "status": "success",
            "result": {
                "current": {
                    "net_profit": float(net_profit),
                    "roi": float(roi),
                    "total_cost": float(cost),
                    "total_benefit": float(benefit),
                    "target_users": int(n_target),
                    "avg_outcome_boost": float(avg_outcome_change)
                },
                "sensitivity": sensitivity_data
            }
        }
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────
# System & Control Room Endpoints
# ──────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    """서버 시작 시: 에이전트 초기화 + Director 아젠다 + 로그 로테이션."""
    db = SessionLocal()
    try:
        if not crud.get_agents(db):
            logger.info("Initializing Genesis Agents...")
            crud.create_agent(db, "theorist-1", "Albert", "Theorist", {"model": "GPT-4o"})
            crud.create_agent(db, "engineer-1", "Tesla", "Engineer", {"sandbox": "Firecracker"})
            crud.create_agent(db, "critic-1", "Kant", "Critic", {"method": "Do-calculus"})
            crud.create_agent(db, "coordinator-1", "Manager", "Coordinator", {"policy": "Themis"})
            
            crud.create_log(db, "coordinator-1", "INFO", "System Boot Sequence Initiated.")
            crud.create_log(db, "theorist-1", "INFO", "Connecting to Knowledge Graph...")

        # Director 아젠다 설정 및 로그 기록
        agenda = _lab_director.get_current_agenda()
        if agenda and "title" in agenda:
            crud.create_log(db, "director", "INFO",
                f"📢 [DIRECTIVE] 금주 연구 주제: '{agenda['title']}' | 카테고리: {agenda.get('category', 'N/A')} | 난이도: {agenda.get('difficulty', 'N/A')}")
            crud.create_log(db, "director", "INFO",
                f"🎯 {agenda.get('description', '')}")
            crud.create_log(db, "coordinator-1", "INFO",
                f"⚡ Director 지시 수신. '{agenda['title']}' 연구 준비 중...")
            crud.create_log(db, "theorist-1", "INFO",
                f"📚 '{agenda['title']}' 관련 선행 연구 탐색 시작...")
            logger.info(f"Director agenda set: {agenda['title']}")

        # Sprint 30: 서버 시작 시 자동 로그 로테이션
        from api.log_rotation import log_rotation
        rotation_stats = log_rotation.rotate()
        if rotation_stats["hot_to_warm"] > 0 or rotation_stats["hot_trimmed"] > 0:
            logger.info(
                "로그 로테이션 완료: Hot→Warm %d건, 트리밍 %d건",
                rotation_stats["hot_to_warm"], rotation_stats["hot_trimmed"]
            )
    finally:
        db.close()

@app.get("/system/agents", response_model=List[schemas.AgentBase])
def get_agents(db: Session = Depends(get_db)):
    return crud.get_agents(db)

@app.get("/system/logs", response_model=List[schemas.SystemLogBase])
def get_system_logs(limit: int = 50, db: Session = Depends(get_db)):
    """최신 시스템 로그 조회 (실시간 스트림용)."""
    return crud.get_logs(db, limit=limit)

@app.post("/system/logs", response_model=schemas.SystemLogBase)
def post_system_log(log: schemas.SystemLogBase, db: Session = Depends(get_db)):
    """에이전트가 직접 로그를 남기는 인터페이스."""
    # Note: 스키마가 id, created_at을 포함하므로 실제 요청용 스키마 분리가 이상적이나,
    # 여기서는 간단히 필드만 취함.
    return crud.create_log(db, log.agent_id, log.level, log.message)

@app.get("/system/graph")
def get_knowledge_graph():
    """지식 그래프 데이터 조회 (NetworkX -> JSON)."""
    from api.graph import kg
    if not kg.initialized:
        kg.initialize_seed_data()
    return kg.get_graph_data()

# ──────────────────────────────────────────────
# Agent Activation (Sprint 12)
# ──────────────────────────────────────────────
@app.post("/system/agents/{agent_id}/activate")
def activate_agent(agent_id: str, db: Session = Depends(get_db)):
    """에이전트를 활성화하여 자율 연구 사이클을 실행합니다."""
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    if agent.status == "WORKING":
        raise HTTPException(status_code=409, detail=f"Agent '{agent_id}' is already working")
    
    # 상태 전이: IDLE → WORKING
    agent.status = "WORKING"
    db.commit()
    crud.create_log(db, agent_id, "INFO", f"Agent '{agent.name}' activated. Status → WORKING")
    
    result = {}
    try:
        cycle_logs = []
        
        if agent.role == "Theorist":
            from api.agents.theorist import run_theorist_cycle
            cycle_logs = run_theorist_cycle()
        elif agent.role == "Engineer":
            from api.agents.engineer import run_engineer_cycle
            cycle_logs = run_engineer_cycle()
        elif agent.role == "Critic":
            from api.agents.critic import run_critic_cycle
            cycle_logs = run_critic_cycle()
        elif agent.role == "Coordinator":
            from api.agents.coordinator import run_coordinator_cycle
            cycle_logs = run_coordinator_cycle()
        
        if cycle_logs:
            # 각 단계를 SystemLog에 기록
            for entry in cycle_logs:
                crud.create_log(db, agent_id, "INFO", f"[{entry['step']}] {entry['message']}")
            
            result = {
                "agent_id": agent_id,
                "role": agent.role,
                "cycle_logs": cycle_logs,
                "status": "COMPLETE",
            }
        else:
            crud.create_log(db, agent_id, "WARNING", f"Agent '{agent.name}' ({agent.role}) returned no logs.")
            result = {
                "agent_id": agent_id,
                "role": agent.role,
                "status": "EMPTY",
                "message": f"{agent.role} 사이클이 결과 없이 종료되었습니다.",
            }
    except Exception as e:
        agent.status = "ERROR"
        db.commit()
        crud.create_log(db, agent_id, "ERROR", f"Execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # 상태 전이: WORKING → IDLE
    agent.status = "IDLE"
    db.commit()
    crud.create_log(db, agent_id, "INFO", f"Agent '{agent.name}' cycle complete. Status → IDLE")
    
    return result

# ──────────────────────────────────────────────
# Agent Evolution (Sprint 15)
# ──────────────────────────────────────────────
@app.post("/system/evolve")
def run_evolution(db: Session = Depends(get_db)):
    """에이전트 성과 평가 및 세대 진화를 실행합니다 (v2)."""
    from api.agents.evolution import run_evolution_cycle
    
    # Manager를 WORKING 상태로
    manager = db.query(models.Agent).filter(models.Agent.role == "Coordinator").first()
    if manager:
        manager.status = "WORKING"
        db.commit()
        crud.create_log(db, manager.id, "INFO", "Evolution Cycle 시작 (v2)")
    
    try:
        evo_logs, evolved_agents = run_evolution_cycle(db)
        
        # 로그 기록
        agent_id = manager.id if manager else None
        for entry in evo_logs:
            crud.create_log(db, agent_id, "INFO", f"[{entry['step']}] {entry['message']}")
        
    except Exception as e:
        if manager:
            manager.status = "ERROR"
            db.commit()
            crud.create_log(db, manager.id, "ERROR", f"Evolution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    if manager:
        manager.status = "IDLE"
        db.commit()
        crud.create_log(db, manager.id, "INFO", "Evolution Cycle 완료")
    
    return {
        "status": "COMPLETE",
        "evolved_agents": [
            {"name": a["name"], "role": a["role"], "generation": a["generation"],
             "specialization": a["specialization"], "parent_score": a["parent_score"]}
            for a in evolved_agents
        ],
        "total_logs": len(evo_logs),
    }

@app.get("/system/evolution/status")
def get_evo_status():
    """진화 시스템 현황 (전략 메모리 + 누적 성과) 조회"""
    from api.agents.evolution import get_evolution_status
    return get_evolution_status()

@app.get("/system/evolution-tree")
def get_evolution_tree(db: Session = Depends(get_db)):
    """세대별 에이전트 진화 트리를 반환합니다."""
    agents = db.query(models.Agent).all()
    
    tree = []
    for agent in agents:
        tree.append({
            "id": agent.id,
            "name": agent.name,
            "role": agent.role,
            "generation": agent.generation or 1,
            "parent_id": agent.parent_id,
            "status": agent.status,
            "config": agent.config,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
        })
    
    return {"agents": tree}

# ──────────────────────────────────────────────
# Research Cycle Dashboard (Sprint 16)
# ──────────────────────────────────────────────
@app.get("/system/cycles")
def get_research_cycles(db: Session = Depends(get_db)):
    """연구 사이클 히스토리를 로그 기반으로 집계합니다."""
    all_logs = db.query(models.SystemLog).order_by(models.SystemLog.created_at.asc()).all()
    
    cycles = []
    current_cycle = None
    
    for log in all_logs:
        msg = log.message or ""
        
        # 사이클 시작 감지
        if "Research Cycle 시작" in msg or "ORCHESTRATE" in msg and "Research Cycle 시작" in msg:
            current_cycle = {
                "id": len(cycles) + 1,
                "started_at": log.created_at.isoformat() if log.created_at else None,
                "ended_at": None,
                "status": "RUNNING",
                "phases": {"theorist": [], "engineer": [], "critic": []},
                "hypotheses": 0,
                "experiments": 0,
                "reviews": 0,
                "verdict": None,
                "logs": [],
            }
        
        if current_cycle:
            current_cycle["logs"].append({
                "message": msg,
                "level": log.level,
                "agent_id": log.agent_id,
                "timestamp": log.created_at.isoformat() if log.created_at else None,
            })
            
            # 가설 카운트
            if "가설 생성" in msg or "Hypothesis" in msg:
                current_cycle["hypotheses"] += 1
            
            # 실험 카운트
            if "실험" in msg and ("설계" in msg or "실행" in msg):
                current_cycle["experiments"] += 1
            
            # 리뷰 카운트
            if "검토" in msg or "Review" in msg or "판정" in msg:
                current_cycle["reviews"] += 1
            
            # 판정 추출
            if "ACCEPT" in msg:
                current_cycle["verdict"] = "ACCEPT"
            elif "REVISE" in msg:
                current_cycle["verdict"] = "REVISE"
            elif "REJECT" in msg:
                current_cycle["verdict"] = "REJECT"
            
            # 사이클 완료 감지
            if "Research Cycle 완료" in msg or "cycle complete" in msg.lower():
                current_cycle["ended_at"] = log.created_at.isoformat() if log.created_at else None
                current_cycle["status"] = "COMPLETE"
                cycles.append(current_cycle)
                current_cycle = None
    
    # 아직 진행 중인 사이클
    if current_cycle:
        current_cycle["status"] = "RUNNING"
        cycles.append(current_cycle)
    
    # 통계 집계
    stats = {
        "total_cycles": len(cycles),
        "completed_cycles": sum(1 for c in cycles if c["status"] == "COMPLETE"),
        "total_hypotheses": sum(c["hypotheses"] for c in cycles),
        "total_experiments": sum(c["experiments"] for c in cycles),
        "total_reviews": sum(c["reviews"] for c in cycles),
        "verdicts": {
            "ACCEPT": sum(1 for c in cycles if c.get("verdict") == "ACCEPT"),
            "REVISE": sum(1 for c in cycles if c.get("verdict") == "REVISE"),
            "REJECT": sum(1 for c in cycles if c.get("verdict") == "REJECT"),
        },
    }
    
    # 로그 상세는 최근 5개 사이클만
    for cycle in cycles:
        cycle["log_count"] = len(cycle["logs"])
        if len(cycles) > 5:
            cycle.pop("logs", None)
    
    return {"cycles": cycles, "stats": stats}

# ──────────────────────────────────────────────
# Auto Research Report (Sprint 17)
# ──────────────────────────────────────────────
@app.get("/system/report")
def get_research_report():
    """자동 연구 보고서를 생성하여 반환합니다."""
    from api.agents.report_generator import generate_report
    return generate_report()

# ──────────────────────────────────────────────
# Academic Forum (Sprint 18)
# ──────────────────────────────────────────────
@app.post("/system/forum")
def run_forum(db: Session = Depends(get_db)):
    """에이전트 간 학술 토론을 실행합니다."""
    from api.agents.forum import run_forum_debate
    
    result = run_forum_debate()
    
    # 로그 기록
    manager = db.query(models.Agent).filter(models.Agent.role == "Coordinator").first()
    agent_id = manager.id if manager else None
    crud.create_log(db, agent_id, "INFO", f"[FORUM] 논제: {result['topic']['topic']}")
    crud.create_log(db, agent_id, "INFO", f"[FORUM] 합의: {result['consensus']['label']}")
    
    return result

# ──────────────────────────────────────────────
# Autopilot Mode (Sprint 19)
# ──────────────────────────────────────────────
@app.post("/system/autopilot/start")
def start_autopilot():
    """Autopilot 자율 순환을 시작합니다."""
    from api.agents.autopilot import autopilot
    return autopilot.start(db_factory=SessionLocal)

@app.post("/system/autopilot/stop")
def stop_autopilot():
    """Autopilot을 정지합니다."""
    from api.agents.autopilot import autopilot
    return autopilot.stop()

@app.get("/system/autopilot/status")
def autopilot_status():
    """Autopilot 상태를 조회합니다."""
    from api.agents.autopilot import autopilot
    return autopilot.get_status()

# ──────────────────────────────────────────────
# Method Registry (Sprint 21)
# ──────────────────────────────────────────────
@app.get("/system/methods")
def get_methods_status():
    """적응형 메서드 레지스트리 현황을 조회합니다."""
    from api.agents.method_registry import method_registry
    return method_registry.get_stats()

# ──────────────────────────────────────────────
# Lab Director & Autonomous Agenda (Sprint 28)
# ──────────────────────────────────────────────
from engine.agents.director import LabDirector

# 프로젝트 루트 기준 경로 사용
_lab_director = LabDirector(knowledge_path=str(ROOT / "data" / "grand_challenges.json"))

@app.get("/system/director/agenda")
def get_director_agenda():
    """연구소장의 현재 연구 아젠다를 조회합니다."""
    return _lab_director.get_current_agenda()

@app.post("/system/director/agenda/next")
def next_director_agenda():
    """연구소장이 새로운 연구 주제를 선택합니다."""
    return _lab_director.set_agenda()

# ──────────────────────────────────────────────
# Sandbox & ConstitutionGuard (Sprint 29)
# ──────────────────────────────────────────────
@app.get("/system/sandbox/status")
def get_sandbox_status():
    """SandboxExecutor 실행 통계를 조회합니다."""
    from engine.sandbox.executor import sandbox
    return sandbox.get_stats()

@app.post("/system/sandbox/reset")
def reset_sandbox_circuit_breaker():
    """회로 차단기를 수동으로 리셋합니다."""
    from engine.sandbox.executor import sandbox
    sandbox.reset_circuit_breaker()
    return {"status": "ok", "message": "회로 차단기가 리셋되었습니다."}

@app.get("/system/constitution/info")
def get_constitution_info():
    """연구 헌법 가드레일 설정 정보를 조회합니다."""
    from api.guards.constitution_guard import ConstitutionGuard
    return {
        "version": "v1.0",
        "rules": {
            "제1조_반증테스트_최소통과": ConstitutionGuard.MIN_REFUTATION_PASSED,
            "제4조_최소방법론수": ConstitutionGuard.MIN_METHODS_COUNT,
            "제5조_표본크기_최소": ConstitutionGuard.SAMPLE_SIZE_MIN,
            "제5조_표본크기_권장": ConstitutionGuard.SAMPLE_SIZE_RECOMMENDED,
            "제12조_메서드집중도_상한": ConstitutionGuard.METHOD_CONCENTRATION_LIMIT,
        },
    }

# ──────────────────────────────────────────────
# Log Rotation & DB Health (Sprint 30)
# ──────────────────────────────────────────────
@app.get("/system/db/status")
def get_db_status():
    """DB 상태 및 로그 로테이션 현황을 조회합니다."""
    from api.log_rotation import log_rotation
    return log_rotation.get_status()

@app.post("/system/db/rotate")
def run_log_rotation():
    """수동으로 로그 로테이션을 실행합니다."""
    from api.log_rotation import log_rotation
    return log_rotation.rotate()

# ──────────────────────────────────────────────
# STEAM Synthetic Data (Sprint 31)
# ──────────────────────────────────────────────
@app.get("/system/steam/dgps")
def get_steam_dgps():
    """사용 가능한 STEAM DGP 템플릿 목록을 조회합니다."""
    from engine.data.steam_generator import steam, DGP_TEMPLATES
    return {
        "available_dgps": steam.available_dgps,
        "templates": {
            name: {
                "name": t.name,
                "grand_challenge_id": t.grand_challenge_id,
                "category": t.category,
                "treatment": t.treatment_name,
                "outcome": t.outcome_name,
                "confounders": t.confounders,
                "moderators": t.moderators,
                "n_default": t.n_default,
                "effect_type": t.effect_type,
            }
            for name, t in DGP_TEMPLATES.items()
        },
    }

@app.post("/system/steam/generate")
def generate_steam_data(dgp_name: str, n: int = 3000, seed: int = 42):
    """STEAM 합성 데이터를 생성하고 요약을 반환합니다."""
    from engine.data.steam_generator import steam
    try:
        data = steam.generate(dgp_name, n=n, seed=seed)
        metrics = steam.evaluate_quality(data)
        return {
            "status": "ok",
            "dgp": dgp_name,
            "sample_size": data.n,
            "ate_true": data.ate_true,
            "columns": list(data.df.columns),
            "head": data.df.head(5).to_dict(orient="records"),
            "quality_metrics": metrics,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/system/agent-registry")
def get_agent_registry():
    """에이전트 중앙 레지스트리를 조회합니다."""
    from api.agent_registry import get_registry_summary
    return get_registry_summary()

# ──────────────────────────────────────────────
# Architect & Hot-Swap (Sprint 33)
# ──────────────────────────────────────────────
@app.get("/system/architect/diagnose")
def run_architect_diagnosis():
    """Architect가 전체 시스템 진단을 실행합니다."""
    from engine.agents.architect import architect
    result = architect.diagnose()
    return result.to_dict()

@app.post("/system/architect/hot-swap")
def hot_swap_module(module_name: str):
    """모듈을 런타임 핫 스왑합니다."""
    from engine.utils.reloader import reloader
    result = reloader.hot_swap(module_name)
    return result.to_dict()

@app.get("/system/architect/backups")
def get_swap_backups():
    """핫 스왑 백업 및 이력을 조회합니다."""
    from engine.utils.reloader import reloader
    return {
        "backups": reloader.list_backups(),
        "swap_history": reloader.get_history(),
    }

# ──────────────────────────────────────────────
# Paper & SaaS (Sprint 34)
# ──────────────────────────────────────────────
@app.post("/system/paper/draft")
def generate_paper_draft(grand_challenge_id: Optional[str] = None, include_latex: bool = False):
    """논문 초안을 자동 생성합니다."""
    from engine.paper.draft_generator import paper_generator
    return paper_generator.generate_draft(grand_challenge_id, include_latex)

@app.get("/system/saas/readiness")
def get_saas_readiness():
    """SaaS 전환 준비도를 평가합니다."""
    from engine.paper.saas_blueprint import saas_blueprint
    return saas_blueprint.assess_readiness()

@app.get("/system/saas/migration-plan")
def get_migration_plan():
    """SaaS 마이그레이션 계획을 조회합니다."""
    from engine.paper.saas_blueprint import saas_blueprint
    return saas_blueprint.get_migration_plan()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=4001, reload=True)

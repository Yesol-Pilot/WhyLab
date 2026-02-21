"""
Autopilot Engine — 완전 자율 연구 순환 시스템
========================================
사용자 개입 없이 Coordinator가 자동으로:
  Research Cycle → Evolution → Forum → Report
를 무한 순환합니다.

[순환 루프]
1. Research Cycle: 가설 생성 → 실험 → 심사
2. Evolution: 성과 평가 → 에이전트 분화
3. Forum: 결과 토론 → 합의 도출
4. Report: 보고서 자동 갱신
5. (대기) → 1로 복귀
"""
import threading
import time
from datetime import datetime, timezone, timedelta

# 한국 표준시 (KST = UTC+9)
KST = timezone(timedelta(hours=9))

def _now_kst() -> datetime:
    """현재 한국 시간 반환"""
    return datetime.now(KST)


class AutopilotEngine:
    """자율 연구 순환 엔진 (싱글턴)"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.running = False
        self.thread = None
        self.cycle_count = 0
        self.current_phase = "IDLE"
        self.history = []
        self.started_at = None
        self.last_cycle_at = None
        self.interval_seconds = 10  # 사이클 간 대기 시간 (8시간 집중 실행용)
    
    def start(self, db_factory):
        """Autopilot 시작"""
        if self.running:
            return {"status": "already_running", "cycle_count": self.cycle_count}
        
        self.running = True
        self.started_at = _now_kst().isoformat()
        self.current_phase = "STARTING"
        self.thread = threading.Thread(
            target=self._run_loop, 
            args=(db_factory,), 
            daemon=True
        )
        self.thread.start()
        return {"status": "started", "started_at": self.started_at}
    
    def stop(self):
        """Autopilot 정지"""
        if not self.running:
            return {"status": "already_stopped"}
        
        self.running = False
        self.current_phase = "STOPPING"
        return {"status": "stopped", "total_cycles": self.cycle_count}
    
    def get_status(self):
        """현재 상태 조회"""
        return {
            "running": self.running,
            "current_phase": self.current_phase,
            "cycle_count": self.cycle_count,
            "started_at": self.started_at,
            "last_cycle_at": self.last_cycle_at,
            "history": self.history[-100:],  # 최근 100건
            "interval_seconds": self.interval_seconds,
        }
    
    def _run_loop(self, db_factory):
        """메인 자율 순환 루프"""
        import logging
        loop_logger = logging.getLogger("whylab.autopilot")
        
        while self.running:
            self.cycle_count += 1
            cycle_start = _now_kst()
            cycle_log = {
                "cycle": self.cycle_count,
                "started_at": cycle_start.isoformat(),
                "phases": [],
            }
            
            try:
                # Phase 1: Research Cycle (에러 격리 — 실패해도 나머지 단계 계속)
                self._update_phase("RESEARCH_CYCLE", cycle_log)
                try:
                    self._run_research_cycle()
                except Exception as e:
                    loop_logger.warning(f"[AUTOPILOT] Research 단계 실패 (스킵): {e}")
                    cycle_log["phases"].append({"phase": "RESEARCH_SKIPPED", "error": str(e)})
                
                # Phase 2: Evolution (독립 DB 세션)
                self._update_phase("EVOLUTION", cycle_log)
                try:
                    db = db_factory()
                    try:
                        self._run_evolution(db)
                    finally:
                        db.close()
                except Exception as e:
                    loop_logger.warning(f"[AUTOPILOT] Evolution 단계 실패 (스킵): {e}")
                    cycle_log["phases"].append({"phase": "EVOLUTION_SKIPPED", "error": str(e)})
                
                # Phase 3: Forum (독립 DB 세션)
                self._update_phase("FORUM", cycle_log)
                try:
                    db = db_factory()
                    try:
                        self._run_forum(db)
                    finally:
                        db.close()
                except Exception as e:
                    loop_logger.warning(f"[AUTOPILOT] Forum 단계 실패 (스킵): {e}")
                    cycle_log["phases"].append({"phase": "FORUM_SKIPPED", "error": str(e)})
                
                # Phase 4: Report (DB 불필요)
                self._update_phase("REPORT_GENERATION", cycle_log)
                self._run_report()
                
                # 완료
                cycle_log["ended_at"] = _now_kst().isoformat()
                cycle_log["status"] = "COMPLETE"
                
            except Exception as e:
                cycle_log["error"] = str(e)
                cycle_log["status"] = "ERROR"
                loop_logger.error(f"[AUTOPILOT] 사이클 {self.cycle_count} 실패: {e}")
            
            self.last_cycle_at = _now_kst().isoformat()
            self.history.append(cycle_log)
            
            # 대기
            if self.running:
                self._update_phase("WAITING", cycle_log)
                self._wait_interruptible(self.interval_seconds)
        
        self.current_phase = "IDLE"
    
    def _update_phase(self, phase, cycle_log):
        """현재 단계 업데이트"""
        self.current_phase = phase
        cycle_log["phases"].append({
            "phase": phase,
            "timestamp": _now_kst().isoformat(),
        })
    
    def _wait_interruptible(self, seconds):
        """중단 가능한 대기"""
        for _ in range(seconds * 10):
            if not self.running:
                return
            time.sleep(0.1)
    
    def _run_research_cycle(self):
        """연구 사이클 실행 (CoordinatorV2 기반)"""
        import logging
        logger = logging.getLogger("whylab.autopilot")
        
        from api.agents.coordinator import CoordinatorV2
        coord = CoordinatorV2()
        result = coord.run_cycle()
        logger.info(f"[AUTOPILOT] 연구 사이클 완료: {result.get('status', 'UNKNOWN')}")
    
    def _run_evolution(self, db):
        """에이전트 진화 + 코드 진화"""
        import logging
        evo_logger = logging.getLogger("whylab.autopilot")

        from api.agents.evolution import run_evolution_cycle
        run_evolution_cycle(db)

        # 코드 진화 (3사이클마다 1회 — Gemini API 비용 절감)
        if self.cycle_count % 3 == 0:
            try:
                from api.agents.code_evolution import code_evolution
                from api.agents.coordinator import CoordinatorV2
                coord = CoordinatorV2()
                data_info = coord._supply_data({}, {"category": "Economy"})
                if data_info:
                    result = code_evolution.evolve(data_info)
                    if result.get("improved"):
                        evo_logger.info(
                            "🧬 코드 진화 성공! RMSE: %.4f → %.4f (%.1f%%)",
                            result["baseline_rmse"], result["new_rmse"],
                            result.get("improvement_pct", 0),
                        )
                    else:
                        evo_logger.info(
                            "🔄 코드 진화 시도: 개선 없음 (Gen %d)",
                            result.get("generation", "?"),
                        )
            except Exception as e:
                evo_logger.warning("코드 진화 실패 (무시): %s", e)
    
    def _run_forum(self, db):
        """학술 토론"""
        import logging
        forum_logger = logging.getLogger("whylab.autopilot")
        from api.agents.forum import run_forum_debate
        
        result = run_forum_debate()
        
        # 로그 기록 (DB 실패 시에도 토론 자체는 성공)
        try:
            from api import crud, models
            manager = db.query(models.Agent).filter(
                models.Agent.role == "Coordinator"
            ).first()
            agent_id = manager.id if manager else None
            crud.create_log(
                db, agent_id, "INFO",
                f"[AUTOPILOT-FORUM] 논제: {result.get('topic', {}).get('topic', '?')} → {result.get('consensus', {}).get('label', '?')}"
            )
        except Exception as e:
            forum_logger.warning(f"[AUTOPILOT] Forum 로그 DB 저장 실패 (무시): {e}")
    
    def _run_report(self):
        """보고서 생성 (갱신)"""
        from api.agents.report_generator import generate_report
        generate_report()  # 보고서는 최신 KG 상태를 반영하므로 자동 갱신됨


# 싱글턴 인스턴스
autopilot = AutopilotEngine()

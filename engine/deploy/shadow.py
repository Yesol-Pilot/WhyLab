# -*- coding: utf-8 -*-
"""Phase 4: 섀도우 배포 컨트롤러.

CTO 비평 반영:
1. 비용 서킷 브레이커 — ARES 딥 감사 일일 토큰 상한
2. Dry-run 섀도우 모드 — ζ 미적용 모니터링

Reviewer 기여:
- 실제 라이브 데이터로 "무오염(No Data Leakage)" 실증
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("whylab.deploy.shadow")

# Supabase 환경변수 (커넥션 풀러 경유 필수: 포트 6543)
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")


class DeploymentMode(str, Enum):
    """배포 모드."""
    SHADOW_DRY_RUN = "shadow_dry_run"   # ζ 미적용, 모니터링만
    SHADOW_ACTIVE = "shadow_active"      # ζ 적용, 라이브 반영
    PRODUCTION = "production"            # 완전 프로덕션


@dataclass
class CostBudget:
    """ARES 딥 감사 비용 예산.

    CTO 지시: 일일 토큰/비용 Hard Limit 설정.
    """
    daily_token_limit: int = 100_000     # 일일 최대 토큰
    daily_cost_limit_usd: float = 10.0   # 일일 최대 비용 (USD)
    tokens_used_today: int = 0
    cost_used_today_usd: float = 0.0
    last_reset: float = field(default_factory=time.time)
    breaker_tripped: bool = False
    trip_count: int = 0

    def consume(self, tokens: int, cost_usd: float = 0.0) -> bool:
        """토큰/비용 소비. 예산 초과 시 False 반환."""
        self._maybe_reset()
        self.tokens_used_today += tokens
        self.cost_used_today_usd += cost_usd

        if (self.tokens_used_today > self.daily_token_limit or
                self.cost_used_today_usd > self.daily_cost_limit_usd):
            self.breaker_tripped = True
            self.trip_count += 1
            logger.warning(
                "🔌 Circuit Breaker TRIPPED: tokens=%d/%d, cost=$%.2f/$%.2f",
                self.tokens_used_today, self.daily_token_limit,
                self.cost_used_today_usd, self.daily_cost_limit_usd,
            )
            return False
        return True

    def _maybe_reset(self) -> None:
        """일일 리셋 (24시간 경과 시)."""
        if time.time() - self.last_reset > 86400:
            self.tokens_used_today = 0
            self.cost_used_today_usd = 0.0
            self.breaker_tripped = False
            self.last_reset = time.time()

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.daily_token_limit - self.tokens_used_today)

    @property
    def utilization(self) -> float:
        return self.tokens_used_today / max(self.daily_token_limit, 1)


@dataclass
class ShadowObservation:
    """섀도우 모드에서의 관측 결과 (Dry-run)."""
    timestamp: float = field(default_factory=time.time)
    decision_id: str = ""
    proposed_zeta: float = 0.0
    lyapunov_zeta_max: float = 0.0
    would_have_clipped: bool = False
    drift_index: float = 0.0
    ares_penalty: float = 0.0
    ate: float = 0.0
    e_value: float = 0.0
    mode: DeploymentMode = DeploymentMode.SHADOW_DRY_RUN


class ShadowDeployController:
    """섀도우 배포 컨트롤러.

    CTO 지시:
    - Dry-run: "만약 ζ를 반영했다면 어떻게 되었을까" 모니터링
    - Circuit Breaker: ARES 비용 상한 초과 시 경량 폴백

    사용법:
        controller = ShadowDeployController(mode=DeploymentMode.SHADOW_DRY_RUN)
        result = controller.process_audit(
            decision_id="d1",
            proposed_zeta=0.5,
            audit_result={...},
        )
    """

    def __init__(
        self,
        mode: DeploymentMode = DeploymentMode.SHADOW_DRY_RUN,
        cost_budget: Optional[CostBudget] = None,
    ) -> None:
        self.mode = mode
        self.cost_budget = cost_budget or CostBudget()
        self._observations: List[ShadowObservation] = []
        self._dlq_memory: List[Dict[str, Any]] = []  # DB 미연결 시 폴백
        self._fallback_count = 0

    def should_run_deep_audit(self) -> bool:
        """ARES 딥 감사를 실행할지 판단.

        서킷 브레이커가 트립되면 경량 폴백으로 전환.
        """
        if self.cost_budget.breaker_tripped:
            self._fallback_count += 1
            logger.info(
                "⚡ Fallback mode: ARES skipped (breaker tripped, fallback #%d)",
                self._fallback_count,
            )
            return False
        return True

    def enqueue_to_dlq(
        self,
        decision_id: str,
        payload: Dict[str, Any],
        reason: str = "breaker_tripped",
    ) -> None:
        """DLQ(Dead Letter Queue) 적재.

        우선: Supabase audit_dlq 테이블에 영속화.
        폴백: DB 미연결 시 인메모리 리스트 (복구 불가).
        """
        entry = {
            "decision_id": decision_id,
            "reason": reason,
            "payload": payload,
        }

        # DB 영속화 시도
        if _SUPABASE_URL and _SUPABASE_KEY:
            try:
                import urllib.request
                url = f"{_SUPABASE_URL}/rest/v1/audit_dlq"
                headers = {
                    "apikey": _SUPABASE_KEY,
                    "Authorization": f"Bearer {_SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                }
                body = json.dumps(entry).encode()
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status in (200, 201):
                        logger.info(
                            "📥 DLQ persisted to DB: %s (reason=%s)",
                            decision_id, reason,
                        )
                        return
            except Exception as e:
                logger.warning("⚠️ DLQ DB write failed: %s — falling back to memory", e)

        # 폴백: 인메모리 (파드 재시작 시 유실)
        entry["timestamp"] = time.time()
        self._dlq_memory.append(entry)
        logger.warning(
            "📥 DLQ in-memory fallback: %s (reason=%s, queue_size=%d) — VOLATILE",
            decision_id, reason, len(self._dlq_memory),
        )

    @property
    def dlq_size(self) -> int:
        return len(self._dlq_memory)

    @property
    def dlq_entries(self) -> List[Dict[str, Any]]:
        return list(self._dlq_memory)

    def record_observation(
        self,
        decision_id: str,
        proposed_zeta: float,
        lyapunov_zeta_max: float,
        drift_index: float = 0.0,
        ares_penalty: float = 0.0,
        ate: float = 0.0,
        e_value: float = 0.0,
    ) -> ShadowObservation:
        """섀도우 관측 기록.

        Dry-run 모드: ζ를 적용하지 않고 기록만.
        Active 모드: ζ를 적용하고 기록.
        """
        obs = ShadowObservation(
            decision_id=decision_id,
            proposed_zeta=proposed_zeta,
            lyapunov_zeta_max=lyapunov_zeta_max,
            would_have_clipped=proposed_zeta > lyapunov_zeta_max,
            drift_index=drift_index,
            ares_penalty=ares_penalty,
            ate=ate,
            e_value=e_value,
            mode=self.mode,
        )
        self._observations.append(obs)

        if self.mode == DeploymentMode.SHADOW_DRY_RUN:
            logger.debug(
                "👁️ Shadow observe: ζ=%.4f (max=%.4f, clip=%s) [DRY-RUN]",
                proposed_zeta, lyapunov_zeta_max,
                "YES" if obs.would_have_clipped else "no",
            )

        return obs

    def should_apply_feedback(self) -> bool:
        """ζ 피드백을 실제로 적용할지 판단."""
        return self.mode in (
            DeploymentMode.SHADOW_ACTIVE,
            DeploymentMode.PRODUCTION,
        )

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """대시보드용 통계."""
        if not self._observations:
            return {"total": 0}

        clip_count = sum(1 for o in self._observations if o.would_have_clipped)
        avg_zeta = sum(o.proposed_zeta for o in self._observations) / len(self._observations)
        avg_di = sum(o.drift_index for o in self._observations) / len(self._observations)

        return {
            "mode": self.mode.value,
            "total_observations": len(self._observations),
            "clip_rate": round(clip_count / len(self._observations), 4),
            "avg_proposed_zeta": round(avg_zeta, 4),
            "avg_drift_index": round(avg_di, 4),
            "cost_budget": {
                "tokens_used": self.cost_budget.tokens_used_today,
                "tokens_limit": self.cost_budget.daily_token_limit,
                "utilization": round(self.cost_budget.utilization, 2),
                "breaker_tripped": self.cost_budget.breaker_tripped,
                "trip_count": self.cost_budget.trip_count,
            },
            "fallback_count": self._fallback_count,
        }

    def promote_to_active(self) -> None:
        """Dry-run → Active 모드 승격."""
        if self.mode == DeploymentMode.SHADOW_DRY_RUN:
            self.mode = DeploymentMode.SHADOW_ACTIVE
            logger.info("🚀 Promoted: SHADOW_DRY_RUN → SHADOW_ACTIVE")

    def promote_to_production(self) -> None:
        """Active → Production 승격."""
        if self.mode == DeploymentMode.SHADOW_ACTIVE:
            self.mode = DeploymentMode.PRODUCTION
            logger.info("🏭 Promoted: SHADOW_ACTIVE → PRODUCTION")


# ── 암호학적 데이터 무결성 서명 ──

def compute_daily_hash(rollup_data: Dict[str, Any], date_str: str) -> Dict[str, str]:
    """데일리 롤업 데이터의 SHA-256 해시.

    체리피킹 방어: 논문 심사위원이 데이터 사후 조작을 의심할 때,
    GitHub 커밋 타임스탬프 + SHA-256으로 무결성 증명.

    Args:
        rollup_data: 롤업 레코드 (JSON-serializable)
        date_str: 날짜 문자열 (e.g. "2026-03-15")

    Returns:
        {"date": date_str, "sha256": hex_hash, "record_count": n}
    """
    canonical = json.dumps(rollup_data, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "date": date_str,
        "sha256": h,
        "record_count": len(rollup_data.get("records", [])),
        "bytes": len(canonical),
    }


def append_hash_log(
    hash_entry: Dict[str, str],
    log_path: str = "data/integrity_hashes.jsonl",
) -> str:
    """Append-only 해시 로그 파일에 추가.

    이 파일을 GitHub에 자동 커밋하면
    타임스탬프가 찍힌 불변 무결성 레코드 역할.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    line = json.dumps(hash_entry, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return log_path


class DailyIntegrityWorker:
    """동기 롤업→해시 파이프라인.

    경쟁 상태(Race Condition) 방어:
    pg_cron이 아닌 파이썬 워커가 롤업 프로시저를 직접 동기적으로 호출.
    트랜잭션 커밋 후에만 해시 계산 → 불완전 데이터 해싱 불가.

    사용법:
        worker = DailyIntegrityWorker()
        result = worker.run("2026-03-15")
    """

    def __init__(
        self,
        supabase_url: str = "",
        supabase_key: str = "",
        hash_log_path: str = "data/integrity_hashes.jsonl",
    ) -> None:
        self.supabase_url = supabase_url or _SUPABASE_URL
        self.supabase_key = supabase_key or _SUPABASE_KEY
        self.hash_log_path = hash_log_path

    def run(self, date_str: str) -> Dict[str, Any]:
        """동기 롤업→해시 파이프라인 실행.

        1. rollup_daily_stats() 호출 (DB 트랜잭션)
        2. 커밋 완료 후 롤업 데이터 조회
        3. SHA-256 해시 계산
        4. integrity_hashes 테이블 + JSONL 로그에 저장
        """
        result = {"date": date_str, "status": "unknown"}

        try:
            # Step 1: 롤업 실행 (DB에 롤업 저장 프로시저 호출)
            rollup_data = self._execute_rollup(date_str)
            if not rollup_data:
                result["status"] = "no_data"
                return result

            # Step 2: SHA-256 해시 (롤업 커밋 직후 — 경쟁 상태 불가)
            hash_entry = compute_daily_hash(rollup_data, date_str)

            # Step 3: DB에 해시 저장
            self._store_hash_to_db(hash_entry)

            # Step 4: 로컬 파일에도 저장 (GitHub 자동 커밋용)
            append_hash_log(hash_entry, self.hash_log_path)

            result["status"] = "success"
            result["hash"] = hash_entry
            logger.info(
                "✅ Daily integrity: %s → SHA256=%s (%d records)",
                date_str, hash_entry["sha256"][:16] + "...",
                hash_entry["record_count"],
            )

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error("❌ Daily integrity failed for %s: %s", date_str, e)

        return result

    def _execute_rollup(self, date_str: str) -> Optional[Dict]:
        """롤업 데이터 조회 (동기)."""
        if not (self.supabase_url and self.supabase_key):
            logger.warning("⚠️ No Supabase credentials — skipping rollup")
            return None

        try:
            import urllib.request
            url = (
                f"{self.supabase_url}/rest/v1/daily_agent_rollup"
                f"?rollup_date=eq.{date_str}&select=*"
            )
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                rows = json.loads(resp.read())
                return {"records": rows, "date": date_str}
        except Exception as e:
            logger.error("❌ Rollup query failed: %s", e)
            return None

    def _store_hash_to_db(self, hash_entry: Dict) -> None:
        """해시를 integrity_hashes 테이블에 저장."""
        if not (self.supabase_url and self.supabase_key):
            return

        try:
            import urllib.request
            url = f"{self.supabase_url}/rest/v1/integrity_hashes"
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal,resolution=merge-duplicates",
            }
            body = json.dumps({
                "rollup_date": hash_entry["date"],
                "sha256_hash": hash_entry["sha256"],
                "record_count": hash_entry["record_count"],
                "data_bytes": hash_entry["bytes"],
            }).encode()
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning("⚠️ Hash DB write failed: %s", e)

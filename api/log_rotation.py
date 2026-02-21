"""
LogRotation — 3계층 데이터 로테이션 (Sprint 30)
=================================================
Autopilot의 무한 자율 순환이 DB를 붕괴시키지 않도록
Hot → Warm → Cold 3계층 로그 관리를 수행합니다.

[설계 문서 §6.2]
- Hot  (whylab.db):        최근 1,000건 / 7일  → 실시간 대시보드
- Warm (whylab_archive.db): 30~90일 보존       → Evolution 성과 추적
- Cold (compressed .jsonl.gz): 90일+ 보존       → 거버넌스 감사
"""
import os
import json
import gzip
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from api.database import SessionLocal

logger = logging.getLogger("whylab.rotation")


class LogRotation:
    """3계층 로그 로테이션 엔진."""

    # ── 보존 정책 ──
    HOT_MAX_ROWS = 1000       # Hot 계층 최대 레코드 수
    HOT_MAX_DAYS = 7          # Hot 계층 최대 보존 기간 (일)
    WARM_MAX_DAYS = 90        # Warm 계층 최대 보존 기간 (일)

    def __init__(self, archive_dir: Optional[str] = None):
        self._archive_dir = Path(archive_dir or "./data/archive")
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._last_rotation: Optional[datetime] = None

    def rotate(self) -> dict:
        """
        전체 로테이션 사이클을 실행합니다.
        
        1. Hot → Warm: 보존 기간 만료된 레코드를 JSONL 파일로 이관
        2. Hot 크기 제한: HOT_MAX_ROWS 초과 시 오래된 레코드 삭제
        3. Cold 압축: WARM_MAX_DAYS 초과 파일은 .gz 압축
        
        Returns:
            dict: 로테이션 결과 통계
        """
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "hot_to_warm": 0,
            "hot_trimmed": 0,
            "warm_to_cold": 0,
            "errors": [],
        }

        db = SessionLocal()
        try:
            # ── Step 1: Hot → Warm (기간 만료 레코드 이관) ──
            cutoff_date = datetime.utcnow() - timedelta(days=self.HOT_MAX_DAYS)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

            # 만료 레코드 조회
            expired_rows = db.execute(
                text(
                    "SELECT id, agent_id, level, message, details, created_at "
                    "FROM system_logs "
                    "WHERE created_at < :cutoff "
                    "ORDER BY created_at ASC"
                ),
                {"cutoff": cutoff_str}
            ).fetchall()

            if expired_rows:
                # JSONL 파일로 저장 (Warm 계층)
                warm_filename = f"logs_{cutoff_date.strftime('%Y%m%d')}.jsonl"
                warm_path = self._archive_dir / warm_filename

                with open(warm_path, "a", encoding="utf-8") as f:
                    for row in expired_rows:
                        record = {
                            "id": row[0],
                            "agent_id": row[1],
                            "level": row[2],
                            "message": row[3],
                            "details": row[4],
                            "created_at": str(row[5]),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Hot에서 삭제
                db.execute(
                    text("DELETE FROM system_logs WHERE created_at < :cutoff"),
                    {"cutoff": cutoff_str}
                )
                db.commit()
                stats["hot_to_warm"] = len(expired_rows)

                logger.info(
                    "Hot→Warm: %d건 이관 → %s",
                    len(expired_rows), warm_path.name
                )

            # ── Step 2: Hot 크기 제한 (최대 행 수 초과 시 트리밍) ──
            row_count = db.execute(
                text("SELECT COUNT(*) FROM system_logs")
            ).scalar()

            if row_count and row_count > self.HOT_MAX_ROWS:
                excess = row_count - self.HOT_MAX_ROWS
                db.execute(
                    text(
                        "DELETE FROM system_logs WHERE id IN ("
                        "  SELECT id FROM system_logs "
                        "  ORDER BY created_at ASC LIMIT :excess"
                        ")"
                    ),
                    {"excess": excess}
                )
                db.commit()
                stats["hot_trimmed"] = excess

                logger.info("Hot 트리밍: %d건 삭제 (최대 %d건 유지)", excess, self.HOT_MAX_ROWS)

            # ── Step 3: Warm → Cold (90일 초과 파일 압축) ──
            cold_cutoff = datetime.utcnow() - timedelta(days=self.WARM_MAX_DAYS)

            for jsonl_file in self._archive_dir.glob("logs_*.jsonl"):
                try:
                    # 파일명에서 날짜 추출 (logs_YYYYMMDD.jsonl)
                    date_str = jsonl_file.stem.replace("logs_", "")
                    file_date = datetime.strptime(date_str, "%Y%m%d")

                    if file_date < cold_cutoff:
                        # .gz 압축
                        gz_path = jsonl_file.with_suffix(".jsonl.gz")
                        with open(jsonl_file, "rb") as f_in:
                            with gzip.open(gz_path, "wb") as f_out:
                                f_out.write(f_in.read())

                        # 원본 삭제
                        jsonl_file.unlink()
                        stats["warm_to_cold"] += 1

                        logger.info("Warm→Cold: %s → %s", jsonl_file.name, gz_path.name)
                except (ValueError, OSError) as e:
                    stats["errors"].append(f"{jsonl_file.name}: {str(e)}")

            # VACUUM으로 디스크 공간 회수 (삭제 후)
            if stats["hot_to_warm"] > 0 or stats["hot_trimmed"] > 0:
                try:
                    db.execute(text("VACUUM;"))
                    logger.info("VACUUM 실행 완료 — 디스크 공간 회수")
                except Exception as e:
                    # WAL 모드에서 VACUUM 실패 시 무시 (비필수)
                    logger.debug("VACUUM 생략: %s", str(e))

        except Exception as e:
            logger.error("로테이션 실행 실패: %s", str(e))
            stats["errors"].append(str(e))
        finally:
            db.close()

        self._last_rotation = datetime.utcnow()
        return stats

    def get_status(self) -> dict:
        """로테이션 현황 조회."""
        db = SessionLocal()
        try:
            hot_count = db.execute(text("SELECT COUNT(*) FROM system_logs")).scalar() or 0
            
            # DB 파일 크기
            db_path = Path("./whylab.db")
            db_size_mb = round(db_path.stat().st_size / (1024 * 1024), 2) if db_path.exists() else 0

            # Warm 파일 수
            warm_files = list(self._archive_dir.glob("logs_*.jsonl"))
            cold_files = list(self._archive_dir.glob("logs_*.jsonl.gz"))

        finally:
            db.close()

        return {
            "hot": {
                "record_count": hot_count,
                "max_records": self.HOT_MAX_ROWS,
                "db_size_mb": db_size_mb,
                "retention_days": self.HOT_MAX_DAYS,
            },
            "warm": {
                "file_count": len(warm_files),
                "retention_days": self.WARM_MAX_DAYS,
            },
            "cold": {
                "compressed_count": len(cold_files),
            },
            "last_rotation": self._last_rotation.isoformat() if self._last_rotation else None,
        }


# 모듈 레벨 싱글턴
log_rotation = LogRotation()

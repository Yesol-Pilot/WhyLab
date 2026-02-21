"""
WhyLab Database Configuration (Sprint 30)
==========================================
- WAL 모드: 읽기/쓰기 병렬 처리 (Autopilot + 대시보드 동시 사용)
- busy_timeout=3000ms: 일시적 쓰기 경합 시 재시도
- 3계층 로테이션: Hot(whylab.db) → Warm(archive) → Cold(압축)
"""
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./whylab.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,  # 연결 상태 사전 확인
)


# ── SQLite PRAGMA 설정 (WAL 모드 + 성능 최적화) ──
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """모든 새 연결에 PRAGMA 설정을 자동 적용합니다."""
    cursor = dbapi_conn.cursor()
    # WAL 모드: 읽기와 쓰기가 서로를 차단하지 않음
    cursor.execute("PRAGMA journal_mode=WAL;")
    # 쓰기 경합 시 3초 대기 후 재시도 (즉시 실패 방지)
    cursor.execute("PRAGMA busy_timeout=3000;")
    # 동기화 수준: NORMAL (안전성 + 성능 균형)
    cursor.execute("PRAGMA synchronous=NORMAL;")
    # 캐시 크기: 약 32MB (기본 2MB에서 확대)
    cursor.execute("PRAGMA cache_size=-32000;")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# 의존성 주입
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


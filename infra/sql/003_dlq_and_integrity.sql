-- 003_dlq_and_integrity.sql
-- DLQ(Dead Letter Queue) 영속화 + 무결성 해시 테이블
-- 
-- 결함 교정:
--   1. 인메모리 DLQ → DB 영속 (파드 재시작/OOM 시 데이터 유실 방지)
--   2. SHA-256 해시를 DB에 영구 저장 (GitHub 이중 백업)

-- ─── DLQ 테이블 ───────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_dlq (
    id              BIGSERIAL PRIMARY KEY,
    decision_id     TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT 'breaker_tripped',
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed       BOOLEAN NOT NULL DEFAULT FALSE,
    processed_at    TIMESTAMPTZ,
    retry_count     INT NOT NULL DEFAULT 0,
    error_message   TEXT
);

-- 미처리 항목 조회 인덱스
CREATE INDEX IF NOT EXISTS idx_dlq_unprocessed 
    ON audit_dlq (processed, created_at) 
    WHERE processed = FALSE;

-- 의사결정 ID 조회
CREATE INDEX IF NOT EXISTS idx_dlq_decision_id 
    ON audit_dlq (decision_id);

COMMENT ON TABLE audit_dlq IS 
    '서킷 브레이커 차단 시 심층 감사 대상 보존. 오프라인 배치 복구용.';


-- ─── 무결성 해시 테이블 ───────────────────────────

CREATE TABLE IF NOT EXISTS integrity_hashes (
    id              BIGSERIAL PRIMARY KEY,
    rollup_date     DATE NOT NULL UNIQUE,
    sha256_hash     TEXT NOT NULL,
    record_count    INT NOT NULL DEFAULT 0,
    data_bytes      INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE integrity_hashes IS
    '일일 롤업 데이터의 SHA-256 해시. 체리피킹 방어. GitHub 이중 저장.';


-- ─── DLQ 배치 처리 함수 ──────────────────────────

CREATE OR REPLACE FUNCTION process_dlq_batch(batch_size INT DEFAULT 100)
RETURNS INT AS $$
DECLARE
    processed_count INT := 0;
BEGIN
    -- 미처리 항목을 처리 완료로 마킹 (실제 ARES 재실행은 앱 레이어)
    UPDATE audit_dlq
    SET processed = TRUE,
        processed_at = NOW(),
        retry_count = retry_count + 1
    WHERE id IN (
        SELECT id FROM audit_dlq
        WHERE processed = FALSE
        ORDER BY created_at ASC
        LIMIT batch_size
        FOR UPDATE SKIP LOCKED
    );
    
    GET DIAGNOSTICS processed_count = ROW_COUNT;
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;

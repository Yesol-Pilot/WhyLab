# -*- coding: utf-8 -*-
"""Phase 11 테스트: API 서버 + 감사 로그 + 버전."""

import pytest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 버전 관리
# ──────────────────────────────────────────────

class TestVersion:
    def test_version_string(self):
        import whylab
        assert whylab.__version__ == "0.2.0"

    def test_pyproject_sync(self):
        """pyproject.toml과 __init__.py 버전 일치 확인."""
        import whylab
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        assert f'version = "{whylab.__version__}"' in pyproject


# ──────────────────────────────────────────────
# 감사 로그
# ──────────────────────────────────────────────

class TestAuditLogger:
    def test_log_analysis(self, tmp_path):
        from engine.audit import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))

        context = {
            "treatment_col": "T",
            "outcome_col": "Y",
            "dataframe": pd.DataFrame({"T": [0, 1], "Y": [1, 2]}),
            "ate": {"point_estimate": 1.5, "ci_lower": 0.5, "ci_upper": 2.5},
            "debate": {"verdict": "CAUSAL", "confidence": 0.85},
            "meta_learners": {"S-Learner": {"ate": 1.4}},
            "quasi_experimental": {"iv": {"ate": 1.3}},
        }

        audit_id = logger.log_analysis(context, execution_time_ms=1234)

        assert audit_id.startswith("AUD-")
        assert len(logger.get_entries()) == 1

        # JSONL 파일 생성 확인
        log_files = list(tmp_path.glob("audit_*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0], "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["verdict"] == "CAUSAL"
        assert entry["treatment"] == "T"
        assert "DML" in entry["methods_used"]
        assert "QE:iv" in entry["methods_used"]

    def test_search(self, tmp_path):
        from engine.audit import AuditLogger

        logger = AuditLogger(log_dir=str(tmp_path))

        # 2건 기록
        logger.log_analysis({
            "treatment_col": "A",
            "outcome_col": "Y",
            "debate": {"verdict": "CAUSAL", "confidence": 0.9},
        })
        logger.log_analysis({
            "treatment_col": "B",
            "outcome_col": "Y",
            "debate": {"verdict": "NOT_CAUSAL", "confidence": 0.7},
        })

        # 검색
        causal_only = logger.search(verdict="CAUSAL")
        assert len(causal_only) == 1
        assert causal_only[0]["treatment"] == "A"

        b_only = logger.search(treatment="B")
        assert len(b_only) == 1
        assert b_only[0]["verdict"] == "NOT_CAUSAL"


# ──────────────────────────────────────────────
# REST API 서버
# ──────────────────────────────────────────────

class TestServer:
    @pytest.fixture(autouse=True)
    def _skip_if_no_fastapi(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def test_health(self):
        from fastapi.testclient import TestClient
        from whylab.server import app

        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["cells"] == 16

    def test_methods(self):
        from fastapi.testclient import TestClient
        from whylab.server import app

        client = TestClient(app)
        res = client.get("/api/v1/methods")
        assert res.status_code == 200
        data = res.json()
        assert data["pipeline_cells"] == 16
        assert "IV (2SLS)" in data["quasi_experimental"]
        assert "Granger Causality" in data["temporal"]

    def test_job_not_found(self):
        from fastapi.testclient import TestClient
        from whylab.server import app

        client = TestClient(app)
        res = client.get("/api/v1/jobs/nonexistent")
        assert res.status_code == 404

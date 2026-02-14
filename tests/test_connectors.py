# -*- coding: utf-8 -*-
"""커넥터 패키지 단위 테스트.

CSVConnector, 팩토리, DataCell 소스 타입 감지를 검증합니다.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.connectors.base import BaseConnector, ConnectorConfig
from engine.connectors.csv_connector import CSVConnector
from engine.connectors.factory import create_connector


# ──────────────────────────────────────────────
# 픽스처: 임시 CSV/Parquet 생성
# ──────────────────────────────────────────────
@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """임시 CSV 파일을 생성합니다."""
    df = pd.DataFrame({
        "treatment": np.random.randint(0, 2, 100),
        "outcome": np.random.randn(100),
        "age": np.random.randint(20, 60, 100),
        "income": np.random.lognormal(8, 0.5, 100),
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """임시 Parquet 파일을 생성합니다."""
    df = pd.DataFrame({
        "treatment": np.random.randint(0, 2, 50),
        "outcome": np.random.randn(50),
        "score": np.random.uniform(0, 1, 50),
    })
    path = tmp_path / "test_data.parquet"
    df.to_parquet(path, index=False)
    return path


# ──────────────────────────────────────────────
# CSVConnector 테스트
# ──────────────────────────────────────────────
class TestCSVConnector:
    """CSV 커넥터 단위 테스트."""

    def test_csv_load_success(self, sample_csv: Path):
        """CSV 파일을 정상적으로 로드합니다."""
        config = ConnectorConfig(
            source_type="csv",
            uri=str(sample_csv),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        with CSVConnector(config) as conn:
            df = conn.fetch()

        assert len(df) == 100
        assert "treatment" in df.columns
        assert "outcome" in df.columns

    def test_parquet_load_success(self, sample_parquet: Path):
        """Parquet 파일을 정상적으로 로드합니다."""
        config = ConnectorConfig(
            source_type="parquet",
            uri=str(sample_parquet),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        with CSVConnector(config) as conn:
            df = conn.fetch()

        assert len(df) == 50
        assert "score" in df.columns

    def test_file_not_found(self, tmp_path: Path):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시킵니다."""
        config = ConnectorConfig(
            source_type="csv",
            uri=str(tmp_path / "nonexistent.csv"),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        with pytest.raises(FileNotFoundError):
            CSVConnector(config).connect()

    def test_missing_treatment_col(self, sample_csv: Path):
        """처치 변수 컬럼이 없으면 ValueError를 발생시킵니다."""
        config = ConnectorConfig(
            source_type="csv",
            uri=str(sample_csv),
            treatment_col="nonexistent_col",
            outcome_col="outcome",
        )
        with pytest.raises(ValueError, match="처치 변수"):
            with CSVConnector(config) as conn:
                conn.fetch()

    def test_fetch_with_meta(self, sample_csv: Path):
        """fetch_with_meta가 올바른 메타데이터를 반환합니다."""
        config = ConnectorConfig(
            source_type="csv",
            uri=str(sample_csv),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        with CSVConnector(config) as conn:
            result = conn.fetch_with_meta()

        assert "dataframe" in result
        assert result["treatment_col"] == "treatment"
        assert result["outcome_col"] == "outcome"
        assert isinstance(result["feature_names"], list)
        # age, income이 자동 추론되어야 함
        assert "age" in result["feature_names"]
        assert "income" in result["feature_names"]

    def test_context_manager(self, sample_csv: Path):
        """Context Manager(with문)가 정상 동작합니다."""
        config = ConnectorConfig(
            source_type="csv",
            uri=str(sample_csv),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        # with문 밖에서도 에러 없이 종료
        connector = CSVConnector(config)
        with connector:
            df = connector.fetch()
            assert len(df) > 0
        # close 후 상태 확인
        assert not connector._connected


# ──────────────────────────────────────────────
# 팩토리 테스트
# ──────────────────────────────────────────────
class TestFactory:
    """커넥터 팩토리 테스트."""

    def test_csv_factory(self, sample_csv: Path):
        """팩토리가 CSV 커넥터를 올바르게 생성합니다."""
        config = ConnectorConfig(
            source_type="csv",
            uri=str(sample_csv),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        connector = create_connector(config)
        assert isinstance(connector, CSVConnector)

    def test_parquet_factory(self, sample_parquet: Path):
        """팩토리가 Parquet에 대해 CSVConnector를 생성합니다."""
        config = ConnectorConfig(
            source_type="parquet",
            uri=str(sample_parquet),
            treatment_col="treatment",
            outcome_col="outcome",
        )
        connector = create_connector(config)
        assert isinstance(connector, CSVConnector)

    def test_unsupported_type(self):
        """지원하지 않는 타입에 ValueError를 발생시킵니다."""
        config = ConnectorConfig(source_type="mongodb")
        with pytest.raises(ValueError, match="지원하지 않는"):
            create_connector(config)

    def test_sql_factory_type(self):
        """팩토리가 SQL 커넥터 클래스를 올바르게 선택합니다."""
        from engine.connectors.sql_connector import SQLConnector

        config = ConnectorConfig(
            source_type="sql",
            uri="sqlite:///test.db",
            treatment_col="t",
            outcome_col="y",
        )
        connector = create_connector(config)
        assert isinstance(connector, SQLConnector)


# ──────────────────────────────────────────────
# DataCell 소스 타입 감지 테스트
# ──────────────────────────────────────────────
class TestSourceTypeDetection:
    """DataCell._detect_source_type 정적 메서드 테스트."""

    def test_csv(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("data.csv") == "csv"

    def test_parquet(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("data.parquet") == "parquet"

    def test_postgresql(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("postgresql://user:pass@host/db") == "postgresql"

    def test_mysql(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("mysql+pymysql://user@host/db") == "mysql"

    def test_sqlite(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("sqlite:///data.db") == "sqlite"

    def test_bigquery(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("bq://project") == "bigquery"

    def test_excel(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("report.xlsx") == "excel"

    def test_tsv(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("data.tsv") == "tsv"

    def test_default_csv(self):
        from engine.cells.data_cell import DataCell
        assert DataCell._detect_source_type("unknown_file.dat") == "csv"

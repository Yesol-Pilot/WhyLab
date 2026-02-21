# -*- coding: utf-8 -*-
"""전체 벤치마크(6종) 실행 스크립트.

IHDP, ACIC, Jobs, TWINS, Criteo, LaLonde-Real 6종 벤치마크에서
모든 메타러너를 평가하고 결과를 JSON + 마크다운으로 저장합니다.

사용법:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --datasets ihdp twins criteo
    python scripts/run_full_benchmark.py --reps 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.cells.benchmark_cell import BenchmarkCell
from engine.config import WhyLabConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ──────────────────────────────────────────────
# 모든 벤치마크 데이터셋 키
# ──────────────────────────────────────────────
ALL_DATASETS = ["ihdp", "acic", "jobs", "twins", "criteo", "lalonde"]


def main():
    parser = argparse.ArgumentParser(description="WhyLab 전체 벤치마크 실행")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"실행할 데이터셋 목록 (기본: {ALL_DATASETS})",
    )
    parser.add_argument(
        "--reps", type=int, default=10,
        help="반복 실험 수 (기본: 10)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="GPU 비활성화",
    )
    args = parser.parse_args()

    # 설정 구성
    config = WhyLabConfig()
    config.benchmark.datasets = args.datasets
    config.benchmark.n_replications = args.reps
    if args.no_gpu:
        config.dml.use_gpu = False

    logger.info("=" * 70)
    logger.info("WhyLab 벤치마크 실행 시작")
    logger.info("  데이터셋: %s", args.datasets)
    logger.info("  반복: %d회", args.reps)
    logger.info("  GPU: %s", "OFF" if args.no_gpu else "ON")
    logger.info("=" * 70)

    # 실행
    cell = BenchmarkCell(config)
    result = cell.execute({})

    # 결과 저장 경로
    output_dir = PROJECT_ROOT / "paper" / "reports" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON 결과 저장
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    json_data = {
        "meta": {
            "datasets": args.datasets,
            "n_replications": args.reps,
            "timestamp": timestamp,
            "gpu": not args.no_gpu,
        },
        "results": result["benchmark_results"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info("JSON 결과 저장: %s", json_path)

    # 마크다운 테이블 저장
    md_path = output_dir / f"benchmark_table_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# WhyLab 벤치마크 결과 ({timestamp})\n\n")
        f.write(f"- 반복: {args.reps}회\n")
        f.write(f"- 데이터셋: {', '.join(args.datasets)}\n\n")
        f.write(result["benchmark_table"])
    logger.info("마크다운 저장: %s", md_path)

    # latest 심볼릭 복사
    latest_json = output_dir / "benchmark_latest.json"
    latest_md = output_dir / "benchmark_latest.md"
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(f"# WhyLab 벤치마크 결과 (최신)\n\n")
        f.write(f"- 반복: {args.reps}회\n")
        f.write(f"- 데이터셋: {', '.join(args.datasets)}\n")
        f.write(f"- 생성: {timestamp}\n\n")
        f.write(result["benchmark_table"])

    logger.info("완료! latest 파일 업데이트됨.")

    # 요약 출력
    print("\n" + "=" * 70)
    print("📊 벤치마크 결과 요약")
    print("=" * 70)
    print(result["benchmark_table"])


if __name__ == "__main__":
    main()

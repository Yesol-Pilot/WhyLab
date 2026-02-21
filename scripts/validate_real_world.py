# -*- coding: utf-8 -*-
"""LaLonde 실데이터 검증 스크립트.

WhyLab v1.0 파이프라인(MAC -> CATE -> Fairness)을 실제 데이터셋에 적용하여 검증합니다.
대상: National Supported Work (NSW) Demonstration dataset (LaLonde, 1986).
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine.data.benchmark_data import BENCHMARK_REGISTRY
from engine.agents.mac_discovery import MACDiscoveryAgent
from engine.cells.meta_learner_cell import TLearner
from engine.cells.fairness_audit_cell import FairnessAuditCell
from engine.config import WhyLabConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def validate_lalonde():
    print(f"\n{'='*60}")
    print("🚀 LaLonde 데이터셋 실데이터 검증 (Real-World Validation)")
    print(f"{'='*60}")

    # 1. 데이터 로드
    if "lalonde" not in BENCHMARK_REGISTRY:
        print("Error: 'lalonde' dataset not found in registry.")
        return

    loader = BENCHMARK_REGISTRY["lalonde"]()
    data = loader.load(seed=42)
    
    # 공변량 이름 (일반적인 LaLonde 데이터셋 변수명 + 잡음 변수)
    feature_names = [
        "age", "education", "black", "hispanic", "married", 
        "nodegre", "re74", "re75", "noise1", "noise2"
    ]
    
    # DataFrame 구성
    df = pd.DataFrame(data.X, columns=feature_names)
    df["treatment"] = data.T
    df["outcome"] = data.Y
    
    print(f"✅ 데이터 로드 완료: N={len(df)}, Features={len(feature_names)}")
    print(f"   Treatment Ratio: {data.T.mean():.2%}")
    print(f"   Outcome Mean: {data.Y.mean():.2f}")

    # 2. MAC 인과 구조 발견 (Discovery)
    # 계산 비용 문제로 일부 샘플/변수만 사용할 수도 있으나, 여기서는 전체 시도
    print("\n🔍 1. MAC Causal Discovery 수행...")
    agent = MACDiscoveryAgent()
    
    # 시간 관계상 re74, re75, treatment, outcome 간의 관계만 확인해볼 수 있음
    # 하지만 전체 변수 넣고 실행 (MAC 내부적으로 PC/GES/LiNGAM 수행)
    discovery_vars = feature_names + ["treatment", "outcome"]
    discovery_data = df[discovery_vars].values
    
    try:
        dag = agent.discover(discovery_data, variable_names=discovery_vars)
        print(f"   -> 발견된 엣지 수: {len(dag.edges)}")
        print(f"   -> 합의 수준(Consensus): {dag.consensus_level:.2%}")
        # 주요 엣지 출력
        print("   -> 주요 인과 경로:")
        for edge in dag.edges[:5]:
            print(f"      {edge.source} -> {edge.target}")
    except Exception as e:
        print(f"   ⚠️ MAC Discovery 실패 (건너뜀): {e}")

    # 3. CATE 추정 (Estimation)
    print("\n📊 2. CATE 추정 (T-Learner XGBoost)...")
    config = WhyLabConfig()
    learner = TLearner(config=config)
    
    learner.fit(data.X, data.T, data.Y)
    cate_pred = learner.predict_cate(data.X)
    
    avg_ate = cate_pred.mean()
    print(f"   -> 추정된 평균 처치 효과 (ATE): {avg_ate:.2f}")
    print(f"      (해석: 직업 훈련이 소득을 평균 ${avg_ate:.2f} 증가시킴)")

    # 4. 공정성 감사 (Fairness Audit)
    print("\n⚖️ 3. 공정성 감사 (Fairness Audit)...")
    fairness_cell = FairnessAuditCell()
    
    # 민감 속성: black, hispanic, married, nodegree (이진 변수들)
    sensitive_attrs = ["black", "hispanic", "married", "nodegree"]
    # 데이터프레임에 이미 있음
    
    audit_results = fairness_cell.audit(
        cate=cate_pred,
        df=df,
        sensitive_attrs=sensitive_attrs
    )
    
    for res in audit_results:
        status = "✅ PASS" if res.is_fair else f"❌ FAIL ({len(res.violations)} violations)"
        print(f"   -> [{res.attribute}] {status}")
        if not res.is_fair:
            for v in res.violations:
                print(f"      - {v}")
        
        # 서브그룹 결과 요약
        groups = [f"{g.group_name}(µ={g.mean_cate:.1f})" for g in res.subgroups]
        print(f"      Subgroups: {', '.join(groups)}")

    print(f"\n{'='*60}")
    print("🎉 실데이터 검증 완료.")
    print(f"{'='*60}")


if __name__ == "__main__":
    validate_lalonde()

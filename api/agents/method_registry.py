"""
Method Registry — 적응형 알고리즘 선택 시스템
============================================
각 에이전트가 사용할 수 있는 메서드 풀을 관리하고,
사이클마다 성공/실패에 따라 가중치를 업데이트하여
더 효과적인 메서드가 자연 선택되도록 합니다.

[핵심 원리: Multi-Armed Bandit with UCB1]
- 각 메서드 = 하나의 Arm
- 보상(reward) = 실험 성공 시 +, 실패/부분 성공 시 ±0
- UCB1 = 평균 보상 + sqrt(2 * ln(전체 시도) / 해당 arm 시도)
- 탐색(exploration)과 착취(exploitation)의 균형 자동 조정
"""
import math
import random
import json
from datetime import datetime
from typing import Optional


class Method:
    """하나의 분석 메서드 (Arm)"""
    
    def __init__(self, name: str, category: str, description: str,
                 params: dict = None, generation: int = 1):
        self.name = name
        self.category = category  # "hypothesis", "experiment", "review"
        self.description = description
        self.params = params or {}
        self.generation = generation
        
        # 밴디트 통계
        self.times_selected = 0
        self.total_reward = 0.0
        self.reward_history = []
        self.created_at = datetime.utcnow().isoformat()
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.times_selected if self.times_selected > 0 else 0.0
    
    def ucb1_score(self, total_trials: int) -> float:
        """UCB1 점수 계산"""
        if self.times_selected == 0:
            return float("inf")  # 아직 안 써본 메서드 우선 탐색
        exploitation = self.avg_reward
        exploration = math.sqrt(2 * math.log(total_trials) / self.times_selected)
        return exploitation + exploration
    
    def record_reward(self, reward: float):
        """보상 기록"""
        self.times_selected += 1
        self.total_reward += reward
        self.reward_history.append({
            "reward": reward,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "params": self.params,
            "generation": self.generation,
            "times_selected": self.times_selected,
            "avg_reward": round(self.avg_reward, 3),
            "total_reward": round(self.total_reward, 3),
        }


class MethodRegistry:
    """에이전트별 적응형 메서드 레지스트리 (싱글턴)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.methods = {}  # {category: [Method, ...]}
        self.selection_log = []
        self._seed_initial_methods()
    
    def _seed_initial_methods(self):
        """초기 메서드 풀 등록"""
        
        # ── Theorist 가설 생성 전략 ──
        self._add("hypothesis", "인과 조절 탐색", 
                  "교란변수가 처리-결과 관계를 조절하는 가설",
                  {"template": "'{confounder}'이(가) '{outcome}'에 미치는 인과 효과는 '{moderator}'에 의해 조절될 수 있다."})
        self._add("hypothesis", "효과 이질성 탐색",
                  "처리 효과가 하위 집단에 따라 다른 가설",
                  {"template": "'{treatment}'의 효과가 '{confounder}' 수준에 따라 이질적(heterogeneous)일 가능성이 있다."})
        self._add("hypothesis", "매개 효과 탐색",
                  "측 관계가 매개 변수를 통해 간접적으로 작용하는 가설",
                  {"template": "'{confounder}'와 '{outcome}' 사이의 관계는 '{treatment}'를 매개로 하여 간접적으로 작용할 수 있다."})
        self._add("hypothesis", "선택 편향 탐지",
                  "교란변수가 처리 배정에 영향을 미치는 가설",
                  {"template": "'{confounder}'이(가) '{treatment}' 배정에 영향을 미쳐 선택 편향을 유발할 수 있다."})
        # Gen 2+ 메서드 (초기에는 비활성이지만 탐색 가능)
        self._add("hypothesis", "비선형 인과 탐색",
                  "변수 간 비선형 인과 관계를 탐색하는 가설",
                  {"template": "'{treatment}'과 '{outcome}' 사이의 관계가 비선형(nonlinear)일 수 있으며, 임계점(threshold) 효과가 존재할 수 있다."}, gen=2)
        self._add("hypothesis", "시간 지연 인과 탐색",
                  "처리 효과의 시간 의존적 지연 패턴을 탐색하는 가설",
                  {"template": "'{treatment}'이 '{outcome}'에 미치는 효과가 시간 지연(lagged effect)을 가지며, '{confounder}'에 의해 지연 기간이 달라질 수 있다."}, gen=2)
        self._add("hypothesis", "교호작용 네트워크 탐색",
                  "다수 교란변수 간의 상호작용 효과를 탐색하는 가설",
                  {"template": "'{confounder}'와 '{moderator}'가 동시에 존재할 때 '{treatment}'의 효과가 개별 교란 효과의 합과 다를 수 있다 (상호작용 효과)."}, gen=2)
        
        # ── Engineer 실험 방법론 ──
        self._add("experiment", "T-Learner (LightGBM)",
                  "두 개의 개별 모델로 처리/대조군 결과를 추정하는 메타러너",
                  {"estimator": "T-Learner", "base_model": "LightGBM", "robustness": 0.7})
        self._add("experiment", "S-Learner (XGBoost)",
                  "단일 모델에 처리 여부를 특성으로 포함하여 추정",
                  {"estimator": "S-Learner", "base_model": "XGBoost", "robustness": 0.65})
        self._add("experiment", "PSM (성향점수매칭)",
                  "성향점수 기반으로 처리/대조군을 매칭하여 ATE 추정",
                  {"estimator": "PSM", "method": "NearestNeighbor", "robustness": 0.6})
        self._add("experiment", "IPW (역확률가중)",
                  "성향점수의 역수로 가중하여 편향을 보정하는 방법",
                  {"estimator": "IPW", "trimming": 0.05, "robustness": 0.6})
        # Gen 2+ 메서드
        self._add("experiment", "DML (이중기계학습)",
                  "Nuisance 파라미터를 ML로 추정한 후 인과 효과를 추정",
                  {"estimator": "DML", "base_model": "RandomForest", "robustness": 0.85}, gen=2)
        self._add("experiment", "Causal Forest",
                  "Random Forest 기반 이질적 처리 효과를 직접 추정",
                  {"estimator": "CausalForest", "n_trees": 500, "robustness": 0.8}, gen=2)
        self._add("experiment", "BART (베이지안 가법 회귀 나무)",
                  "베이지안 방법론으로 비선형 처리 효과를 추정",
                  {"estimator": "BART", "n_trees": 200, "robustness": 0.75}, gen=2)
        self._add("experiment", "DR-Learner (이중강건추정)",
                  "IPW와 Outcome Regression을 결합하여 이중 강건 추정",
                  {"estimator": "DR-Learner", "base_model": "ElasticNet", "robustness": 0.9}, gen=2)
        
        # ── Critic 평가 기준 ──
        self._add("review", "표본 규모 검정",
                  "표본 크기가 통계적 검정력을 확보하기에 충분한지 평가",
                  {"min_n": 500, "severity": "WARNING"})
        self._add("review", "신뢰구간 포함 검정",
                  "ATE의 95% 신뢰구간이 0을 포함하는지 확인",
                  {"null_value": 0, "severity": "CRITICAL"})
        self._add("review", "이질성 p-value 검정",
                  "서브그룹 이질성 검정의 p-value 임계값 기반 평가",
                  {"p_threshold": 0.05, "severity": "WARNING"})
        # Gen 2+ 기준
        self._add("review", "E-value 민감도 분석",
                  "미관찰 교란에 대한 E-value 기반 민감도 평가",
                  {"e_value_threshold": 2.0, "severity": "IMPORTANT"}, gen=2)
        self._add("review", "다중 비교 보정 검정",
                  "Bonferroni/BH 보정을 적용한 다중 비교 보정 평가",
                  {"correction": "BH", "severity": "WARNING"}, gen=2)
        self._add("review", "외부 타당도 검정",
                  "연구 결과가 다른 모집단에도 일반화 가능한지 평가",
                  {"criteria": ["population_diversity", "context_similarity"], "severity": "IMPORTANT"}, gen=2)
        self._add("review", "재현성 체크리스트",
                  "실험의 재현 가능성을 코드/데이터/환경 관점에서 평가",
                  {"items": ["code_available", "data_accessible", "env_reproducible"], "severity": "WARNING"}, gen=2)
    
    def _add(self, category: str, name: str, desc: str, params: dict, gen: int = 1):
        """메서드 등록"""
        method = Method(name, category, desc, params, gen)
        self.methods.setdefault(category, []).append(method)
    
    def select_method(self, category: str, generation: int = 1) -> Method:
        """
        UCB1 기반으로 최적의 메서드를 선택합니다.
        
        세대가 높을수록 더 고급 메서드에 접근 가능합니다.
        """
        available = [
            m for m in self.methods.get(category, [])
            if m.generation <= generation
        ]
        
        if not available:
            raise ValueError(f"카테고리 '{category}'에 사용 가능한 메서드가 없습니다.")
        
        total_trials = sum(m.times_selected for m in available) + 1
        
        # UCB1 점수 기반 선택
        best = max(available, key=lambda m: m.ucb1_score(total_trials))
        
        self.selection_log.append({
            "category": category,
            "selected": best.name,
            "ucb1_score": round(best.ucb1_score(total_trials), 3),
            "generation": generation,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return best
    
    def select_methods(self, category: str, count: int = 1, generation: int = 1) -> list[Method]:
        """여러 메서드를 UCB1 순 상위 N개 선택"""
        available = [
            m for m in self.methods.get(category, [])
            if m.generation <= generation
        ]
        
        if not available:
            return []
        
        total_trials = sum(m.times_selected for m in available) + 1
        ranked = sorted(available, key=lambda m: m.ucb1_score(total_trials), reverse=True)
        return ranked[:count]
    
    def reward_method(self, method_name: str, category: str, reward: float):
        """메서드에 보상을 기록합니다."""
        for m in self.methods.get(category, []):
            if m.name == method_name:
                m.record_reward(reward)
                return
    
    def get_stats(self) -> dict:
        """전체 레지스트리 통계"""
        stats = {}
        for category, methods in self.methods.items():
            stats[category] = {
                "total_methods": len(methods),
                "methods": sorted(
                    [m.to_dict() for m in methods],
                    key=lambda x: x["avg_reward"],
                    reverse=True
                ),
                "total_selections": sum(m.times_selected for m in methods),
            }
        return {
            "categories": stats,
            "last_10_selections": self.selection_log[-10:],
        }
    
    def discover_new_method(self, category: str, base_method: Method) -> Optional[Method]:
        """
        기존 메서드를 기반으로 새로운 변형 메서드를 발견합니다.
        조건: 기존 메서드의 avg_reward ≥ 0.7 이고 10회 이상 사용
        """
        if base_method.avg_reward < 0.7 or base_method.times_selected < 10:
            return None
        
        # 변형 이름 생성
        variant_count = sum(
            1 for m in self.methods.get(category, [])
            if base_method.name in m.name
        )
        new_name = f"{base_method.name} v{variant_count + 1}"
        
        # 이미 존재하면 스킵
        for m in self.methods.get(category, []):
            if m.name == new_name:
                return None
        
        # 파라미터 변형
        new_params = dict(base_method.params)
        if "robustness" in new_params:
            new_params["robustness"] = min(1.0, new_params["robustness"] + 0.05)
        
        new_method = Method(
            name=new_name,
            category=category,
            description=f"{base_method.description} (적응형 변형)",
            params=new_params,
            generation=base_method.generation + 1,
        )
        self.methods[category].append(new_method)
        return new_method


# 싱글턴 인스턴스
method_registry = MethodRegistry()

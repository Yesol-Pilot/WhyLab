"""
[검증] E2E Pipeline 1-Cycle Test
================================
Coordinator v2의 전체 7단계 파이프라인을 1회 실행하여
실제 데이터 흐름과 에이전트 연동을 검증합니다.

[점검 항목]
1. Director: Agenda 선택
2. STEAM: 데이터 생성 + CSV 저장
3. Theorist: 가설 생성 (LLM Fallback)
4. Engineer: 가설 → 실험 설계 → Sandbox 실행
5. Critic: 실험 결과 판정 (Gemini/Rule)
6. KG: 결과 반영 (Edge 업데이트)
"""
import sys
import os
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Windows 콘솔 인코딩 이슈 방지
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
logger = logging.getLogger("whylab.e2e_test")

# 프로젝트 루트 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_e2e_test():
    logger.info("🚀 E2E 1-Cycle 검증 시작...")
    
    try:
        from api.agents.coordinator import CoordinatorV2
        
        # 1. Coordinator 초기화
        coord = CoordinatorV2()
        logger.info("✅ CoordinatorV2 초기화 완료")
        
        # 2. 사이클 실행
        result = coord.run_cycle()
        
        # 3. 결과 검증
        logger.info("-" * 50)
        logger.info(f"🏁 사이클 종료 | 상태: {result['status']}")
        
        stages = result.get("stages", [])
        logger.info(f"📂 실행 단계: {len(stages)}개")
        for s in stages:
            logger.info(f"  [{s['stage']}] {s['message']}")
            
        # STEAM 데이터 경로 확인
        data_stage = next((s for s in stages if s["stage"] == "DATA"), None)
        if data_stage and "data_path" in result:
            logger.info(f"✅ STEAM 데이터 경로: {result['data_path']}")
        
        # 최종 상태 확인
        if result["status"] in ["completed", "COMPLETE"]:
            logger.info("✅ E2E 검증 성공!")
        else:
            logger.error(f"❌ E2E 검증 실패: {result['status']}")
            
    except Exception as e:
        logger.exception(f"❌ E2E 검증 중 치명적 오류: {e}")
        raise

if __name__ == "__main__":
    run_e2e_test()

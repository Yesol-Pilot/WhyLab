import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell

def test_data_cell():
    print("🚀 DataCell 테스트 시작...")
    
    # 설정 초기화
    config = WhyLabConfig()
    
    # DataCell 실행
    cell = DataCell(config)
    
    # 실행 (scenario='C'로 호출해도 로컬 CSV가 있으면 무시하고 CSV 로드해야 함)
    result = cell.execute({"scenario": "C"})
    
    df = result["dataframe"]
    print(f"✅ 데이터 로드 성공: {len(df)} rows")
    print(f"📄 컬럼 목록: {list(df.columns)}")
    print(f"📁 시나리오 확인: {result['scenario']}")
    
    if "treatment" in df.columns:
        print("🎯 treatment 컬럼 확인됨")
    else:
        print("❌ treatment 컬럼 누락")

if __name__ == "__main__":
    try:
        test_data_cell()
    except Exception as e:
        print(f"🔥 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

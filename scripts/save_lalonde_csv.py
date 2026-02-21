import sys
import os
import pandas as pd

# 프로젝트 루트 경로 추가 (절대 경로)
sys.path.append(r"d:\00.test\PAPER\WhyLab")

from engine.data.benchmark_data import LaLondeRealLoader

if __name__ == "__main__":
    print("Generating LaLonde test dataset...")
    try:
        loader = LaLondeRealLoader()
        data = loader.load(n=2000)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # DataFrame 생성
    df = pd.DataFrame(data.X, columns=data.feature_names)
    df['treatment'] = data.T
    df['outcome'] = data.Y
    
    # 저장
    try:
        os.makedirs("data", exist_ok=True)
        output_path = "data/lalonde_test.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ LaLonde dataset saved to {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

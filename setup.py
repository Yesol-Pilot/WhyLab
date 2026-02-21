import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_step(msg):
    print(f"\n🚀 [Setup] {msg}")

def check_python_version():
    print_step("Python 버전 확인")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print(f"❌ Python 3.9 이상이 필요합니다. (현재: {sys.version})")
        sys.exit(1)
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")

def create_directories():
    print_step("필수 디렉토리 생성")
    dirs = [
        "engine/data",
        "logs",
        "db",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Directory: {d}")

def check_env_file():
    print_step(".env 파일 확인")
    if not os.path.exists(".env"):
        print("⚠️ .env 파일이 없습니다. .env.example을 복사합니다.")
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ .env 생성 완료 (API Key 설정 필요)")
        else:
            with open(".env", "w", encoding="utf-8") as f:
                f.write("GEMINI_API_KEY=\n")
            print("✅ 빈 .env 파일 생성 완료. GEMINI_API_KEY를 입력해주세요.")
    else:
        print("✅ .env 파일 존재함")

def create_test_data():
    print_step("테스트 데이터 확인")
    data_path = Path("engine/data/test_scenario.csv")
    if not data_path.exists():
        print("⚠️ 테스트 데이터가 없습니다. 생성합니다.")
        content = """user_id,income,age,credit_score,app_usage_time,consumption,treatment,outcome
1,5000,30,700,10,2000,1,0
2,6000,35,750,15,2500,0,1
3,4000,25,650,5,1500,1,1
4,7000,40,800,20,3000,0,0
5,5500,32,720,12,2200,1,0"""
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ {data_path} 생성 완료")
    else:
        print(f"✅ {data_path} 존재함")

def main():
    print("="*50)
    print("      WhyLab Autonomous Setup Wizard      ")
    print("="*50)
    
    check_python_version()
    create_directories()
    check_env_file()
    create_test_data()
    
    print("\n🎉 설정 완료! 시스템을 실행할 준비가 되었습니다.")

if __name__ == "__main__":
    main()

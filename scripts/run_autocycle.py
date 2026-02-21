import requests
import time
import json
import sys

BASE_URL = "http://localhost:4001"

def run_autocycle():
    print("🚀 [Step 1] Autopilot 시작 요청...")
    try:
        res = requests.post(f"{BASE_URL}/system/autopilot/start")
        print(f"   -> Status: {res.status_code}, Response: {res.json()}")
    except Exception as e:
        print(f"❌ Autopilot 시작 실패: {e}")
        return

    print("\n⏳ [Step 2] 20초간 연구 사이클 진행 중 (Theorist -> Engineer -> Critic)...")
    
    start_time = time.time()
    seen_logs = set()
    
    while time.time() - start_time < 20:
        try:
            # 로그 폴링
            res = requests.get(f"{BASE_URL}/system/logs?limit=5")
            if res.ok:
                logs = res.json()
                for log in logs:
                    log_id = log.get("id")
                    if log_id not in seen_logs:
                        seen_logs.add(log_id)
                        agent = log.get("agent_id", "System")
                        msg = log.get("message", "")
                        ts = log.get("timestamp", "").split("T")[-1][:8]
                        print(f"   [{ts}] {agent}: {msg}")
            
            # 메서드 현황 폴링 (간헐적)
            if int(time.time()) % 5 == 0:
                m_res = requests.get(f"{BASE_URL}/system/methods")
                if m_res.ok:
                    data = m_res.json()
                    total = sum(len(c["methods"]) for c in data["categories"].values())
                    # print(f"   [System] Method Registry Active: {total} methods loaded.")
                    
        except Exception as e:
            pass
        
        time.sleep(1)

    print("\n🛑 [Step 3] Autopilot 종료 요청...")
    try:
        requests.post(f"{BASE_URL}/system/autopilot/stop")
        print("   -> Autopilot 종료 완료.")
    except:
        print("   -> 종료 요청 중 에러 발생 (이미 종료되었을 수 있음)")

    print("\n✅ 검증 완료. 위 로그에서 'Gemini', 'Engine', 'ATE' 등의 키워드를 확인하세요.")

if __name__ == "__main__":
    run_autocycle()

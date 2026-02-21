import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:4001"

def check_status():
    print("🔍 [System Status Check]")
    print("-" * 50)
    
    # 1. Methods Status
    try:
        res = requests.get(f"{BASE_URL}/system/methods")
        data = res.json()
        print(f"✅ Method Registry: Online")
        for cat, info in data.get("categories", {}).items():
            print(f"   - {cat}: {len(info.get('methods', []))} methods active")
            if info.get("last_10_selections"):
                last = info["last_10_selections"][-1]
                print(f"     -> Latest: {last.get('selected')} (Gen {last.get('generation')})")
    except:
        print("❌ Method Registry: Offline or Error")

    print("-" * 50)

    # 2. Recent Cycles
    try:
        res = requests.get(f"{BASE_URL}/system/cycles")
        data = res.json()
        cycles = data.get("cycles", [])
        print(f"✅ Research Cycles: {len(cycles)} recorded")
        for c in cycles[-3:]:
            print(f"   [Cycle #{c['id']}]")
            print(f"   🧠 Hypothesis: {c.get('hypothesis', {}).get('text', '')[:60]}...")
            print(f"   🧪 Experiment: ATE={c.get('experiment', {}).get('ate', '?')} (Method: {c.get('experiment', {}).get('method')})")
            print(f"   ⚖️ Critic Verdict: {c.get('critic', {}).get('verdict')}")
    except:
        print("❌ Research Cycles: Offline or Error")

    print("-" * 50)

    # 3. Recent Logs (Autopilot activity)
    try:
        res = requests.get(f"{BASE_URL}/system/logs?limit=10")
        logs = res.json()
        print(f"✅ Recent System Logs (Latest 10)")
        for log in logs:
            print(f"   [{log['timestamp'].split('T')[-1][:8]}] {log['agent_id']}: {log['message']}")
    except:
        print("❌ System Logs: Offline or Error")

if __name__ == "__main__":
    check_status()

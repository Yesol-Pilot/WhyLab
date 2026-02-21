"""WhyLab Autopilot 성과 분석 스크립트"""
import sys, os, json, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze():
    # === 1. KG 상태 ===
    from api.graph import kg
    if not kg.initialized:
        kg.initialize_seed_data()
    
    print("=" * 60)
    print("1. 지식 그래프 (KG) 상태")
    print("=" * 60)
    stats = kg.get_stats()
    print(f"  노드: {stats.get('total_nodes', 0)}")
    print(f"  엣지: {stats.get('total_edges', 0)}")
    print(f"  검증된 엣지: {stats.get('verified_edges', 0)}")
    print(f"  가설: {stats.get('total_hypotheses', 0)}")
    print(f"  실험: {stats.get('total_experiments', 0)}")
    
    print("\n  [노드 목록]")
    for n, d in kg.graph.nodes(data=True):
        cat = d.get("category", "?")
        print(f"    [{cat}] {n}")
    
    print("\n  [엣지 목록]")
    for u, v, d in kg.graph.edges(data=True):
        rel = d.get("relation", "?")
        w = d.get("weight", 0)
        verified = "V" if d.get("verified") else " "
        hyp = d.get("hypothesis_id", "")
        ate = d.get("ate", "")
        verdict = d.get("verdict", "")
        method = d.get("method", "")
        print(f"    [{verified}] {u} --({rel})--> {v}  w={w}")
        if hyp:
            print(f"        가설={hyp} 방법={method} ATE={ate} 판정={verdict}")
    
    # === 2. 서버 Autopilot 상태 ===
    print("\n" + "=" * 60)
    print("2. Autopilot 사이클 이력")
    print("=" * 60)
    try:
        r = requests.get("http://localhost:4001/system/autopilot/status", timeout=10)
        d = r.json()
        total = d.get("cycle_count", 0)
        hist = d.get("history", [])
        complete = sum(1 for h in hist if h.get("status") == "COMPLETE")
        errors = sum(1 for h in hist if h.get("status") == "ERROR")
        
        print(f"  총 사이클: {total}")
        print(f"  COMPLETE: {complete}")
        print(f"  ERROR: {errors}")
        if complete + errors > 0:
            print(f"  성공률: {round(complete/(complete+errors)*100, 1)}%")
        
        print("\n  [사이클 상세]")
        for h in hist:
            c = h.get("cycle", "?")
            s = h.get("status", "?")
            sa = str(h.get("started_at", ""))[:19]
            ea = str(h.get("ended_at", ""))[:19]
            skips = [p for p in h.get("phases", []) if "SKIP" in str(p.get("phase", ""))]
            skip_info = ""
            if skips:
                skip_info = " SKIP:" + ",".join(p.get("phase", "") for p in skips)
            err = ""
            if h.get("error"):
                err = f" ERR={str(h['error'])[:60]}"
            print(f"    #{c}: {s} [{sa} ~ {ea}]{skip_info}{err}")
    except Exception as e:
        print(f"  서버 접속 실패: {e}")
    
    # === 3. Coordinator 실제 결과 ===
    print("\n" + "=" * 60)
    print("3. 연구 결과 분석 (직접 1사이클 실행)")
    print("=" * 60)
    try:
        from api.agents.coordinator import CoordinatorV2
        coord = CoordinatorV2()
        result = coord.run_cycle()
        
        print(f"  사이클: {result.get('cycle_id', '?')}")
        print(f"  상태: {result.get('status', '?')}")
        print(f"  소요: {result.get('elapsed_seconds', '?')}초")
        
        hyp = result.get("hypothesis", {})
        if hyp:
            print(f"\n  [가설]")
            print(f"    텍스트: {hyp.get('text', '?')[:120]}")
            print(f"    소스: {hyp.get('hypothesis_source', '?')}")
            print(f"    변수: IV={hyp.get('iv', '?')}, DV={hyp.get('dv', '?')}")
        
        exp = result.get("experiment", {})
        if exp:
            print(f"\n  [실험]")
            print(f"    방법: {exp.get('method', '?')}")
            print(f"    ATE: {exp.get('ate', '?')}")
            print(f"    p-value: {exp.get('p_value', '?')}")
            print(f"    통계적 유의: {exp.get('significant', '?')}")
        
        verd = result.get("verdict", {})
        if verd:
            print(f"\n  [판정]")
            print(f"    행동: {verd.get('action', '?')}")
            print(f"    소스: {verd.get('source', '?')}")
            for k, v in verd.items():
                if k not in ("action", "source") and v:
                    print(f"    {k}: {str(v)[:100]}")
        
        metrics = result.get("metrics", {})
        if metrics:
            print(f"\n  [지표]")
            for k, v in metrics.items():
                print(f"    {k}: {v}")
                
    except Exception as e:
        print(f"  사이클 실행 실패: {e}")
    
    # === 4. KG 변화 확인 (사이클 후) ===
    print("\n" + "=" * 60)
    print("4. 사이클 후 KG 상태")
    print("=" * 60)
    stats2 = kg.get_stats()
    print(f"  노드: {stats.get('total_nodes', 0)} -> {stats2.get('total_nodes', 0)}")
    print(f"  엣지: {stats.get('total_edges', 0)} -> {stats2.get('total_edges', 0)}")
    print(f"  검증된 엣지: {stats.get('verified_edges', 0)} -> {stats2.get('verified_edges', 0)}")

if __name__ == "__main__":
    analyze()

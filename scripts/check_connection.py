import requests
import time
import socket

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def check_url(url, desc):
    try:
        start = time.time()
        res = requests.get(url, timeout=5)
        elapsed = time.time() - start
        print(f"[{'✅' if res.ok else '❌'}] {desc} ({url}) -> Status: {res.status_code} ({elapsed:.2f}s)")
        return res.ok
    except Exception as e:
        print(f"[❌] {desc} ({url}) -> Error: {e}")
        return False

def main():
    print("🕵️ System Connection Diagnostic")
    print("="*40)
    
    # 1. Port Check
    print("\n1. Port Availability Check")
    fe_port = check_port("localhost", 4000)
    be_port = check_port("localhost", 4001)
    print(f"   - Frontend (4000): {'OPEN' if fe_port else 'CLOSED'}")
    print(f"   - Backend (4001): {'OPEN' if be_port else 'CLOSED'}")
    
    # 2. Backend Check
    print("\n2. Backend API Check")
    if be_port:
        check_url("http://localhost:4001/", "Root")
        check_url("http://localhost:4001/system/methods", "Methods API")
        check_url("http://localhost:4001/docs", "Swagger UI")
    
    # 3. Frontend Check
    print("\n3. Frontend Dashboard Check")
    if fe_port:
        check_url("http://localhost:4000/", "Root Page")
        check_url("http://localhost:4000/dashboard", "Dashboard Page")
        # Next.js may return 404 but status 200 on some conf
        
    print("\nDiagnostic Complete.")

if __name__ == "__main__":
    main()

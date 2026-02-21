# -*- coding: utf-8 -*-
"""시스템 리소스 확인 스크립트."""
import os
import sys

def check_memory():
    try:
        import psutil
    except ImportError:
        print("psutil 미설치. pip install psutil 필요")
        return
    
    m = psutil.virtual_memory()
    print(f"=== RAM ===")
    print(f"  전체: {m.total/1024**3:.1f} GB")
    print(f"  사용: {m.used/1024**3:.1f} GB ({m.percent}%)")
    print(f"  가용: {m.available/1024**3:.1f} GB")
    print()
    
    # 상위 메모리 소비 프로세스
    procs = []
    for p in psutil.process_iter(['name', 'memory_info', 'pid']):
        try:
            mi = p.info.get('memory_info')
            if mi:
                procs.append((p.info['name'], mi.rss, p.info['pid']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    procs.sort(key=lambda x: x[1], reverse=True)
    print("=== 상위 메모리 소비 프로세스 ===")
    for name, rss, pid in procs[:15]:
        print(f"  {name:35s} {rss/1024**2:>8.0f} MB  (PID {pid})")

def check_gpu():
    try:
        import torch
        print(f"\n=== GPU ===")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")
        else:
            print(f"  CUDA compiled: {torch.version.cuda}")
    except ImportError:
        print("\n  PyTorch 미설치")

if __name__ == "__main__":
    check_memory()
    check_gpu()

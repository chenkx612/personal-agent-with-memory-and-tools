import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":
    # 先输出提示，再触发重依赖的 import，改善启动体感
    print("Initializing Personal Agent...", flush=True)
    from interfaces.cli import main
    main()

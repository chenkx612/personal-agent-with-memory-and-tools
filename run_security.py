import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from personal_agent.security import main

if __name__ == "__main__":
    main()

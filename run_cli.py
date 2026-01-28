import sys
import os

# Add src to python path so we can import personal_agent
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from personal_agent.interfaces.cli import main

if __name__ == "__main__":
    main()

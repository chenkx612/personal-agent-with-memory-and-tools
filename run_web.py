import sys
import os
from streamlit.web import cli as stcli

def main():
    # Add src to python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(project_root, "src"))

    # Path to the web interface file
    web_interface_path = os.path.join(project_root, "src", "personal_agent", "interfaces", "web.py")

    # Construct the command line arguments for streamlit
    sys.argv = ["streamlit", "run", web_interface_path]
    
    # Run streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()

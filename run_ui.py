#!/usr/bin/env python
"""Runner script for JARVIS Streamlit UI."""
import subprocess
import sys
from pathlib import Path

def main():
    """Run Streamlit app."""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#1e3c72",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6",
        "--theme.textColor", "#262730",
    ]
    
    print("Lancement de JARVIS Web UI...")
    print("   URL: http://localhost:8501")
    print("   Appuyez sur Ctrl+C pour arreter")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nArret de JARVIS Web UI")
    except subprocess.CalledProcessError as e:
        print(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
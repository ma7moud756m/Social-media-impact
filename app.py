"""Root entry point proxy for Streamlit dashboard.

Running `streamlit run app.py` will launch `dashboard/app.py`.
"""
import runpy
from pathlib import Path

target_file = Path(__file__).resolve().parent / "dashboard" / "app.py"
runpy.run_path(str(target_file), run_name="__main__")

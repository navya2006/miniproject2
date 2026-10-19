"""Run the full MP2 pipeline in order:  python run_all.py"""
import runpy, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
for step in ["01_clean.py", "02_models.py", "03_diagnostics.py", "04_robustness.py", "05_figures.py", "06_descriptives.py"]:
    print(f"\n=== {step} ===")
    runpy.run_path(str(here / step), run_name="__main__")
print("\nDone. Tables -> output_tables/, figures -> figures/, cleaned data -> data/")

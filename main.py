#!/usr/bin/env python3
"""MP2 - Salary determinants at HCL Technologies.  Run the whole pipeline.

    python main.py data/ITEmp.xlsx -o outputs            # analysis + figures + table images + Excel workbook
    python main.py data/ITEmp.xlsx -o outputs --no-visuals   # numbers only (results.json, CSV tables)

Stages run in this order; each lives in its own module under mp2/.
"""
import argparse
import os
import sys
import time

from mp2.context import Context
from mp2 import (data_cleaning, descriptives, models, battery, effects, diagnostics, robustness,
                 cleaning_sensitivity, storage)

ANALYSIS = [("Data cleaning", data_cleaning), ("Descriptives", descriptives), ("Models M1-M6", models),
            ("Quadratic / interaction battery", battery), ("Effects and hypotheses", effects),
            ("Diagnostics", diagnostics), ("Robustness", robustness),
            ("Cleaning-threshold sensitivity", cleaning_sensitivity), ("Save results", storage)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="path to ITEmp.xlsx")
    ap.add_argument("-o", "--out", default="outputs", help="output directory (default: outputs)")
    ap.add_argument("--no-visuals", action="store_true", help="skip figures, table images and the Excel workbook")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.source):
        ap.error(f"input file not found: {a.source}")
    return a


def main(argv=None):
    a = parse_args(argv)
    ctx = Context(a.source, a.out)
    stages = list(ANALYSIS)
    if not a.no_visuals:
        from mp2 import plots_data, plots_models, plots_diagnostics, plots_robustness, dashboard, table_render, export_excel
        stages += [("Data figures", plots_data), ("Model figures", plots_models),
                   ("Diagnostic figures", plots_diagnostics), ("Robustness figures", plots_robustness),
                   ("Summary dashboard", dashboard), ("Table images", table_render), ("Excel workbook", export_excel)]
    for name, mod in stages:
        t0 = time.time()
        mod.run(ctx)
        print(f"  [{time.time() - t0:5.1f}s] {name}")
    print(f"done. N = {ctx.N} | N family sample = {len(ctx.S4)} | outputs in {a.out}/")


if __name__ == "__main__":
    sys.exit(main())

"""Shared paths, variable lists and helper functions for the MP2 project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TABLES = ROOT / "output_tables"
FIGS = ROOT / "figures"
for p in (TABLES, FIGS):
    p.mkdir(exist_ok=True)

RAW_XLSX = DATA / "ITEmp.xlsx"
CLEAN_CSV = DATA / "ITEmp_Cleaned.csv"
COMPLETE_CSV = DATA / "ITEmp_CompleteCases.csv"

CONT = ["sal", "edu", "exper", "ten", "dep", "momed", "daded", "sib"]
BINARY = ["nonprev", "female", "married", "metro", "norcen", "south", "west",
          "itreq", "itprog", "itqual", "itarch", "itai", "itclient", "itrd",
          "ittech", "wrkhm"]
ALL_VARS = CONT + BINARY  # 24 analytic variables, in the order of Table 1
# Table 1 lists them in this order (dictionary order)
TABLE1_ORDER = ["sal", "edu", "exper", "ten", "dep", "momed", "daded", "sib",
                "nonprev", "female", "married", "metro", "norcen", "south",
                "west", "itreq", "itprog", "itqual", "itarch", "itai",
                "itclient", "itrd", "ittech", "wrkhm"]

# Plausible ranges taken from the data dictionary (anything outside -> missing)
RANGES = {
    "sal": (0.01, 100.0),   # USD per hour; must be positive (log is taken)
    "edu": (1, 30),          # years of formal education
    "exper": (0, 70),
    "ten": (0, 60),
    "dep": (0, 20),
    "momed": (0, 30),
    "daded": (0, 30),
    "sib": (0, 20),
}

# Model blocks
HUMAN = ["edu", "exper", "ten"]
DEMOG = ["female", "married", "dep", "nonprev"]
GEO = ["metro", "norcen", "south", "west"]
JOB = ["itreq", "itprog", "itqual", "itarch", "itai", "itclient", "itrd",
       "ittech", "wrkhm"]
WORKGROUPS = ["itreq", "itprog", "itqual", "itarch"]
DIVISIONS = ["itrd", "ittech"]
FAMILY = ["momed", "daded", "sib"]

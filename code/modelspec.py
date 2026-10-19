"""Model construction helpers (centring, design matrices, model definitions)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import *

CENTRED = ["edu", "exper", "ten", "dep", "momed", "daded", "sib"]


def prepare(df):
    """Add mean-centred variables, their squares and the interaction terms."""
    d = df.copy()
    for v in CENTRED:
        d[f"c_{v}"] = d[v] - d[v].mean()
    for v in ["edu", "exper", "ten"]:
        d[f"c_{v}_sq"] = d[f"c_{v}"] ** 2
    d["c_edu:female"] = d["c_edu"] * d["female"]
    d["married:female"] = d["married"] * d["female"]
    d["c_exper:c_edu"] = d["c_exper"] * d["c_edu"]
    d["metro:c_edu"] = d["metro"] * d["c_edu"]
    return d


HUMAN_C = ["c_edu", "c_exper", "c_ten"]
DEMOG_C = ["female", "married", "c_dep", "nonprev"]
GEO_C = GEO
JOB_C = JOB
FAM_C = ["c_momed", "c_daded", "c_sib"]
QUAD_C = ["c_edu_sq", "c_exper_sq", "c_ten_sq"]
INTER_C = ["c_edu:female", "married:female", "c_exper:c_edu", "metro:c_edu"]

SPECS = {
    "M1 Human capital": HUMAN_C,
    "M2 + Demographics": HUMAN_C + DEMOG_C,
    "M3 + Geography": HUMAN_C + DEMOG_C + GEO_C,
    "M4 + Job function": HUMAN_C + DEMOG_C + GEO_C + JOB_C,
    "M5 Full main effects": HUMAN_C + DEMOG_C + GEO_C + JOB_C + FAM_C,
    "M6 + Quadratics": HUMAN_C + DEMOG_C + GEO_C + JOB_C + FAM_C + QUAD_C,
    "M7 + Interactions": HUMAN_C + DEMOG_C + GEO_C + JOB_C + FAM_C + QUAD_C + INTER_C,
}


def fit(d, cols, y="lnsal", cov=None):
    X = sm.add_constant(d[cols])
    m = sm.OLS(d[y], X)
    return m.fit(cov_type=cov) if cov else m.fit()


def load_clean():
    return prepare(pd.read_csv(CLEAN_CSV))

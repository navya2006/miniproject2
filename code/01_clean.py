"""Step 1 - data cleaning.

Rules (see report Section 2):
  * non-numeric text in a numeric field          -> missing
  * value outside the plausible/dictionary range -> missing
  * binary indicator not in {0, 1}               -> missing
  * missing continuous -> column median, missing binary -> column mode
No observations are added or dropped. A complete-case file is also saved (M10).
"""
import numpy as np
import pandas as pd
from common import *

raw = pd.read_excel(RAW_XLSX, sheet_name=0)
raw = raw.rename(columns=str.strip)
assert len(raw) == 526, len(raw)

num = raw[["id"] + ALL_VARS].copy()
audit = []
clean = raw[["id"]].copy()
for v in ALL_VARS:
    s = raw[v]
    was_missing = s.isna()
    coerced = pd.to_numeric(s, errors="coerce")
    nonnum = (~was_missing) & coerced.isna()
    if v in BINARY:
        oor = coerced.notna() & ~coerced.isin([0, 1])
    else:
        lo, hi = RANGES[v]
        oor = coerced.notna() & ((coerced < lo) | (coerced > hi))
    coerced = coerced.mask(oor)
    clean[v] = coerced
    audit.append(dict(variable=v, role="dependent" if v == "sal" else
                      ("binary" if v in BINARY else "continuous"),
                      missing_in_raw=int(was_missing.sum()),
                      non_numeric=int(nonnum.sum()),
                      out_of_range=int(oor.sum()),
                      total_set_to_missing=int(clean[v].isna().sum())))
audit = pd.DataFrame(audit).set_index("variable").loc[TABLE1_ORDER].reset_index()
audit.to_csv(TABLES / "table1_cleaning_audit.csv", index=False)

# complete-case sample (model M10)
complete = clean.dropna().copy()
complete.to_csv(COMPLETE_CSV, index=False)

# imputation
imputed = clean.copy()
n_imputed = 0
for v in ALL_VARS:
    m = imputed[v].isna()
    n_imputed += int(m.sum())
    fill = imputed[v].median() if v in CONT else imputed[v].mode().iloc[0]
    imputed.loc[m, v] = fill

# derived variables
imputed["lnsal"] = np.log(imputed["sal"])
for v in ["exper", "ten", "edu"]:
    imputed[f"{v}_sq"] = imputed[v] ** 2
imputed.to_csv(CLEAN_CSV, index=False)

tot = audit[["missing_in_raw", "non_numeric", "out_of_range"]].sum()
print(audit.to_string(index=False))
print("totals:", tot.to_dict(), "| cells imputed:", n_imputed,
      f"({n_imputed / (526 * 24):.2%} of {526 * 24})", "| complete cases:", len(complete))

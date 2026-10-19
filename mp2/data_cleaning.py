"""Stage A - load ITEmp.xlsx and clean it with four logged, rule-based steps.

The Excel file itself is never modified. Every action is written to cleaning_log.csv.
  Rule 1  type coercion          non-numeric / blank cells -> missing
  Rule 2  range checks           impossible values -> missing
  Rule 3  block logic            a missing dummy inside a mutually exclusive block is 0 if another dummy is 1
  Rule 4  consistency flag       tenure > experience is flagged, not altered
"""
import os
import numpy as np
import pandas as pd
from .config import BIN, REGION, FUNC, DIV, FAM, CORE_VARS, RANGE


def _is_blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _category(issue):
    if issue.startswith("blank"):
        return "blank cell"
    if issue.startswith("non-numeric"):
        return "non-numeric text"
    if issue.startswith("impossible"):
        return "impossible / implausible value"
    if issue.startswith("dummy"):
        return "dummy outside {0,1}"
    return "recovered via block logic"


def run(ctx):
    raw = pd.read_excel(ctx.src, sheet_name="Salary", dtype=object)
    raw.columns = [str(c).strip() for c in raw.columns]      # defensive: strip header whitespace
    assert raw["id"].is_unique
    log = []

    def note(i, var, raw_val, issue, action):
        log.append(dict(id=int(i), variable=var, raw_value="" if _is_blank(raw_val) else raw_val,
                        issue=issue, action=action))

    df = pd.DataFrame({"id": pd.to_numeric(raw["id"]).astype(int)})
    for c in [c for c in raw.columns if c != "id"]:                      # Rule 1
        num = pd.to_numeric(raw[c], errors="coerce")
        for i in raw.index[num.isna()]:
            v = raw.at[i, c]
            note(raw.at[i, "id"], c, v, "blank cell" if _is_blank(v) else "non-numeric text", "set to missing")
        df[c] = num

    for c, (lo, hi, why) in RANGE.items():                               # Rule 2
        bad = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
        for i in df.index[bad]:
            note(df.at[i, "id"], c, df.at[i, c], f"impossible/implausible value ({why})", "set to missing")
        df.loc[bad, c] = np.nan
    for c in BIN:
        bad = df[c].notna() & ~df[c].isin([0, 1])
        for i in df.index[bad]:
            note(df.at[i, "id"], c, df.at[i, c], "dummy not in {0,1}", "set to missing")
        df.loc[bad, c] = np.nan

    valid_rows = df[BIN].notna().all(axis=1)                             # Rule 3
    for name, blk in [("region", REGION), ("function", FUNC), ("division", DIV)]:
        assert df.loc[valid_rows, blk].sum(axis=1).max() <= 1, name
        for i in df.index:
            vals = df.loc[i, blk]
            if vals.isna().any() and vals.fillna(0).sum() == 1:
                for c in blk:
                    if np.isnan(df.at[i, c]):
                        df.at[i, c] = 0
                        note(df.at[i, "id"], c, raw.at[i, c], "invalid/blank dummy inside a mutually exclusive block",
                             "recovered as 0 (another dummy of the same block equals 1)")
    df["ten_gt_exper"] = ((df["ten"] > df["exper"]) & df["ten"].notna() & df["exper"].notna()).astype(int)   # Rule 4

    df["complete_core"] = df[CORE_VARS].notna().all(axis=1)
    df["complete_fam"] = df[CORE_VARS + FAM].notna().all(axis=1)
    df["lsal"] = np.log(df["sal"])
    for v in ("edu", "exper", "ten", "dep"):
        df[v + "2"] = df[v] ** 2

    lg = pd.DataFrame(log).sort_values(["id", "variable"])
    lg["cat"] = lg["issue"].map(_category)
    df.to_csv(os.path.join(ctx.out, "ITEmp_cleaned.csv"), index=False)
    lg.drop(columns="cat").to_csv(os.path.join(ctx.out, "cleaning_log.csv"), index=False)

    S = df[df.complete_core].copy().reset_index(drop=True)
    S4 = df[df.complete_fam].copy().reset_index(drop=True)
    N = len(S)
    ctx.raw, ctx.df, ctx.log, ctx.S, ctx.S4, ctx.N = raw, df, lg, S, S4, N

    recovered = lg.action.str.startswith("recovered")
    clean = dict(
        n_raw=len(raw), n_cols=len(raw.columns), n_flag_cells=int((lg.cat != "recovered via block logic").sum()),
        n_recovered=int(recovered.sum()), n_rows_flagged=int(lg.id.nunique()),
        by_cat=lg[lg.cat != "recovered via block logic"].groupby("cat").size().to_dict(),
        by_var=lg[~recovered].groupby("variable").size().to_dict(),
        n_final=N, n_final_fam=len(S4), n_ten_gt_exper=int(S.ten_gt_exper.sum()),
        sal_removed=lg[lg.variable == "sal"][["id", "raw_value", "issue"]].to_dict(orient="records"),
        edu_removed=lg[lg.variable == "edu"][["id", "raw_value"]].to_dict(orient="records"))

    # sample waterfall: first reason for exclusion, in this order
    order = [("sal (salary missing/invalid)", ["sal"]), ("edu", ["edu"]), ("exper", ["exper"]),
             ("metro / region dummies", ["metro"] + REGION), ("IT function / division dummies", FUNC + DIV)]
    keep = pd.Series(True, index=df.index)
    wf = []
    for lab, vs in order:
        miss = df[vs].isna().any(axis=1) & keep
        wf.append((lab, int(miss.sum())))
        keep &= ~miss
    assert int(keep.sum()) == N
    clean["waterfall"] = wf
    clean["fam_loss"] = int(N - len(S4))
    raw_sal = pd.to_numeric(raw["sal"], errors="coerce")
    clean["raw_sal_max"], clean["raw_sal_min"] = float(raw_sal.max()), float(raw_sal.min())
    clean["sal_second_max"] = float(raw_sal.sort_values().dropna().iloc[-4])
    clean["sal_max_kept"] = float(S.sal.max())
    ctx.R["clean"] = clean

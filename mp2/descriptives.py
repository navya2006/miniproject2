"""Stage B - descriptive statistics, group means and the pre-drawn plotting jitter."""
import numpy as np
from scipy import stats
from .config import BIN, FAM, FUNC


def run(ctx):
    S, S4, R, N = ctx.S, ctx.S4, ctx.R, ctx.N
    cont = ["sal", "lsal", "edu", "exper", "ten", "dep"]
    R["desc_cont"] = S[cont].agg(["count", "mean", "std", "min", "median", "max"]).T.reset_index() \
        .rename(columns={"index": "var"}).to_dict(orient="records")
    R["desc_fam"] = S4[FAM].agg(["count", "mean", "std", "min", "median", "max"]).T.reset_index() \
        .rename(columns={"index": "var"}).to_dict(orient="records")
    R["desc_bin"] = {c: dict(n1=int(S[c].sum()), share=float(S[c].mean())) for c in BIN}
    R["skew"] = dict(sal=float(stats.skew(S.sal)), lsal=float(stats.skew(S.lsal)),
                     kurt_sal=float(stats.kurtosis(S.sal, fisher=False)),
                     kurt_lsal=float(stats.kurtosis(S.lsal, fisher=False)))
    R["corr"] = S[["lsal", "edu", "exper", "ten", "dep"]].corr().round(3).to_dict()
    R["corr_ex_ten"] = float(S.exper.corr(S.ten))
    R["desc_by_gender"] = S.groupby("female").agg(
        n=("sal", "size"), mean_sal=("sal", "mean"), med_sal=("sal", "median"), mean_lsal=("lsal", "mean"),
        edu=("edu", "mean"), exper=("exper", "mean"), ten=("ten", "mean")).reset_index().to_dict(orient="records")
    R["corr_fam"] = S4[["lsal", "edu", "exper", "ten", "dep", "momed", "daded", "sib"]].corr().round(3).to_dict()

    # gender x marital-status means
    gm = S.groupby(["female", "married"]).lsal.agg(["mean", "std", "count"]).reset_index()
    gm["ci"] = 1.96 * gm["std"] / np.sqrt(gm["count"])
    R["gender_marital_means"] = gm.to_dict(orient="records")

    # readable job-group labels and their mean log salaries
    S["function"] = np.select([S[c] == 1 for c in FUNC], ["Requirements", "Programming", "Quality assurance",
                                                          "Architecture", "AI applications", "Client-only"],
                              default="None of the six")
    S["division"] = np.select([S.itrd == 1, S.ittech == 1], ["R&D", "Tech support"], default="Neither")
    S["wfh"] = np.where(S.wrkhm == 1, "Works from home", "Not primarily home")
    grp = {}
    for col in ["function", "division", "wfh"]:
        g = S.groupby(col).lsal.agg(["mean", "std", "count"]).sort_values("mean")
        g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
        grp[col] = g.reset_index().to_dict(orient="records")
    R["group_means"] = grp

    R["n_by_group"] = {c: int(S[c].sum()) for c in ["female", "nonprev", "married", "metro", "wrkhm", "itai", "itarch", "itrd"]}
    R["sample_means"] = {c: float(S[c].mean()) for c in ["edu", "exper", "ten"]}
    R["sal_stats_sample"] = dict(mean=float(S.sal.mean()), median=float(S.sal.median()))

    # Jitter for the discrete variables in scatter plots is drawn here, in the original order
    # (edu, then ten), so that plotting has no side effects on the random stream used by the bootstraps.
    ctx.jitter = {"edu": ctx.rng.uniform(-0.25, 0.25, N), "ten": ctx.rng.uniform(-0.25, 0.25, N)}

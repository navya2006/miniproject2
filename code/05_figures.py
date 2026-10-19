"""Step 5 - all report figures (Figures 1-9) written to ../figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from common import *
from modelspec import *

sns.set_theme(style="whitegrid")
d = load_clean()
m7 = fit(d, SPECS["M7 + Interactions"])


def save(fig, name):
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Fig 1 - distribution of salary, levels and logs
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
sns.histplot(d["sal"], bins=30, kde=True, ax=ax[0], color="#2f6db5"); ax[0].set(title="Average hourly salary (raw)", xlabel="USD per hour")
sns.histplot(d["lnsal"], bins=30, kde=True, ax=ax[1], color="#2e8b5f"); ax[1].set(title="log(average hourly salary)", xlabel="log USD per hour")
fig.suptitle("Figure 1. The dependent variable is right-skewed; the log transform restores near-symmetry")
save(fig, "fig1_salary_distribution.png")

# Fig 2 - correlation matrix
fig, ax = plt.subplots(figsize=(11, 9.5))
cols = ["lnsal"] + [c for c in ALL_VARS if c != "sal"]
sns.heatmap(d[cols].corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 6}, linewidths=.3, ax=ax, cbar_kws={"shrink": .75})
ax.set_title("Figure 2. Correlation matrix (lnsal = log average hourly salary)")
save(fig, "fig2_correlation_matrix.png")

# Fig 3 - function and division differentials
wg = d[WORKGROUPS + ["itarch"]].copy()
d["workgroup"] = np.select([d.itreq == 1, d.itprog == 1, d.itqual == 1, d.itarch == 1],
                           ["Requirements", "Programming", "Quality ass.", "Architecture"], "Other / reference")
order = ["Requirements", "Programming", "Quality ass.", "Architecture", "Other / reference"]
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
sns.boxplot(data=d, x="workgroup", y="sal", order=order, hue="workgroup", palette="Blues_r", legend=False, ax=ax[0])
ax[0].set(title="Salary by IT work-group", xlabel="", ylabel="USD per hour"); ax[0].tick_params(axis="x", rotation=20)
sns.boxplot(data=d, x="itrd", y="sal", hue="itrd", palette="Greens", legend=False, ax=ax[1])
ax[1].set(title="Salary by IT R&D", ylabel="USD per hour", xticks=[0, 1], xticklabels=["Non R&D", "IT R&D"])
sns.boxplot(data=d, x="ittech", y="sal", hue="ittech", palette="Oranges", legend=False, ax=ax[2])
ax[2].set(title="Salary by IT tech support", ylabel="USD per hour", xticks=[0, 1], xticklabels=["No", "Yes"])
fig.suptitle("Figure 3. Function and division differentials in pay")
save(fig, "fig3_function_division.png")

# Fig 4 - geography
d["province"] = np.select([d.norcen == 1, d.south == 1, d.west == 1], ["North-central", "South", "West"], "East (reference)")
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
sns.boxplot(data=d, x="province", y="sal", order=["East (reference)", "North-central", "South", "West"],
            hue="province", palette="Purples", legend=False, ax=ax[0])
ax[0].set(title="Salary by province", xlabel="", ylabel="USD per hour"); ax[0].tick_params(axis="x", rotation=15)
sns.boxplot(data=d, x="metro", y="sal", hue="metro", palette="Reds", legend=False, ax=ax[1])
ax[1].set(title="Salary by metro area", ylabel="USD per hour", xticks=[0, 1], xticklabels=["Non-metro", "Metro"])
fig.suptitle("Figure 4. Geographic pay differentials")
save(fig, "fig4_geography.png")

# Fig 5 - salary vs human capital (linear solid, quadratic dashed)
fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
for a, (v, t) in zip(ax, [("edu", "Years of education"), ("exper", "Years of experience"), ("ten", "Years of tenure")]):
    sns.regplot(data=d, x=v, y="sal", order=1, scatter_kws=dict(s=10, alpha=.45, color="#4a5568"), line_kws=dict(color="#c53030"), ax=a)
    if v != "edu":
        sns.regplot(data=d, x=v, y="sal", order=2, scatter=False, line_kws=dict(color="#2b6cb0", linestyle="--"), ax=a)
    a.set(title=t, ylabel="USD per hour")
fig.suptitle("Figure 5. Salary vs human-capital stock\nsolid = linear fit; dashed = quadratic fit where shown")
save(fig, "fig5_human_capital.png")

# Fig 6 - violins by group
fig, ax = plt.subplots(1, 4, figsize=(16, 5.5), sharey=True)
for a, (v, labs) in zip(ax, [("female", ["Male", "Female"]), ("married", ["Unmarried", "Married"]),
                              ("wrkhm", ["On-site", "Work from home"]), ("itai", ["No AI work", "AI applications"])]):
    sns.violinplot(data=d, x=v, y="sal", hue=v, palette="Set2", legend=False, inner="quartile", cut=0, ax=a)
    a.set(xlabel="", ylabel="USD per hour" if v == "female" else "", xticks=[0, 1], xticklabels=labs)
fig.suptitle("Figure 6. Salary distributions across demographic and work-mode groups")
save(fig, "fig6_group_violins.png")

# Fig 7 - experience vs tenure; salary by experience band
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].scatter(d["exper"], d["ten"], s=14, alpha=.45, color="#2f6db5", edgecolor="none")
ax[0].set(title="Experience vs tenure", xlabel="Years of experience", ylabel="Years of tenure")
bands = pd.cut(d["exper"], [0, 5, 10, 15, 20, 25, 30, 55], labels=["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-55"])
bm = d.groupby(bands, observed=True)["sal"].mean()
ax[1].plot(bm.index.astype(str), bm.values, "o-", color="#2e8b5f", lw=2, ms=8)
ax[1].set(title="Mean salary by experience band", xlabel="Years of experience", ylabel="Mean USD per hour"); ax[1].tick_params(axis="x", rotation=25)
fig.suptitle("Figure 7. The experience-tenure collinearity and the non-linear experience-salary profile")
save(fig, "fig7_experience_tenure.png")

# Fig 8 - residual diagnostics for M7
infl = m7.get_influence()
cook = infl.cooks_distance[0]
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].scatter(m7.fittedvalues, m7.resid, s=10, alpha=.5, color="#2f6db5"); ax[0, 0].axhline(0, color="red")
ax[0, 0].set(title="Residuals vs fitted", xlabel="Fitted log salary", ylabel="Residual")
sm.qqplot(m7.resid, line="45", ax=ax[0, 1], markerfacecolor="#2e8b5f", markeredgecolor="#2e8b5f", markersize=4)
ax[0, 1].set_title("Normal Q-Q plot")
ax[1, 0].scatter(m7.fittedvalues, np.sqrt(np.abs(infl.resid_studentized_internal)), s=10, alpha=.5, color="#805ad5")
ax[1, 0].set(title="Scale-location", xlabel="Fitted log salary", ylabel="sqrt(|standardised residual|)")
ax[1, 1].vlines(range(len(cook)), 0, cook, lw=.8); ax[1, 1].axhline(4 / len(cook), color="#c53030", ls="--", lw=1, label="4/n")
ax[1, 1].set(title="Cook's distance", xlabel="Observation", ylabel="Cook's D"); ax[1, 1].legend(fontsize=8)
fig.suptitle("Figure 8. Assumption diagnostics for the preferred specification (M7)")
fig.tight_layout(rect=[0, 0, 1, .97])
save(fig, "fig8_residual_diagnostics.png")

# Fig 9 - interaction effects, other regressors at their sample means
means = {c: d[c].mean() for c in SPECS["M7 + Interactions"]}
base_raw = {"edu": d.edu.mean(), "exper": d.exper.mean()}


def predict(edu=None, exper=None, female=None):
    x = dict(means)
    e = base_raw["edu"] if edu is None else edu
    x_ex = base_raw["exper"] if exper is None else exper
    ce, cx = e - d.edu.mean(), x_ex - d.exper.mean()
    f = means["female"] if female is None else female
    x.update({"c_edu": ce, "c_exper": cx, "c_edu_sq": ce ** 2, "c_exper_sq": cx ** 2, "female": f,
              "c_edu:female": ce * f, "married:female": means["married"] * f, "c_exper:c_edu": cx * ce, "metro:c_edu": means["metro"] * ce})
    row = pd.Series(x)[m7.params.index.drop("const")]
    return float(np.exp(m7.params["const"] + (row * m7.params.drop("const")).sum()))


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ev = np.linspace(8, 17, 100)
ax[0].plot(ev, [predict(edu=e, female=0) for e in ev], color="#c53030", lw=2, label="Men")
ax[0].plot(ev, [predict(edu=e, female=1) for e in ev], color="#2b6cb0", lw=2, label="Women")
ax[0].set(title="Education premium by gender (H11)", xlabel="Years of education", ylabel="Predicted salary (USD/hour)"); ax[0].legend()
xv = np.linspace(d.exper.quantile(.02), d.exper.quantile(.98), 100)
lo, hi = d.edu.quantile(.25), d.edu.quantile(.75)
ax[1].plot(xv, [predict(exper=x, edu=lo) for x in xv], color="#c53030", lw=2, label="Lower education (P25)")
ax[1].plot(xv, [predict(exper=x, edu=hi) for x in xv], color="#2b6cb0", lw=2, label="Higher education (P75)")
ax[1].set(title="Experience profile by education (H12)", xlabel="Years of experience", ylabel="Predicted salary (USD/hour)"); ax[1].legend()
fig.suptitle("Figure 9. Estimated interaction effects (other regressors at their sample means)")
save(fig, "fig9_interaction_effects.png")
print("figures written:", sorted(p.name for p in FIGS.glob("*.png")))

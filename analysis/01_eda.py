"""Exploratory data analysis: summary statistics, group default rates,
distributions, and correlations. Writes tables and figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BILL_COLS, ENGINEERED, FIGURES, MONEY_COLS, PAY_COLS,
                    PAYAMT_COLS, TABLES, TARGET, load_clean)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})
GREY, DARK, ACCENT = "#9a9a9a", "#2b2b2b", "#1f5f8b"

df = load_clean()

# ---- Univariate summary ----------------------------------------------------
num_cols = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS + ENGINEERED
summary = df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
summary["skew"] = df[num_cols].skew()
summary = summary.round(3)
summary.to_csv(TABLES / "eda_summary_numeric.csv")

cat_rows = []
for c in ["SEX", "EDUCATION", "MARRIAGE", "AGE_BAND", TARGET]:
    vc = df[c].value_counts().sort_index()
    for lvl, n in vc.items():
        cat_rows.append({"variable": c, "level": lvl, "count": int(n), "share": round(n / len(df), 4)})
pd.DataFrame(cat_rows).to_csv(TABLES / "eda_summary_categorical.csv", index=False)

# ---- Bivariate: default rate by group --------------------------------------
rows = []
for c in ["SEX", "AGE_BAND", "EDUCATION", "MARRIAGE", "PAY_0"]:
    g = df.groupby(c)[TARGET].agg(["count", "mean"]).reset_index()
    for _, r in g.iterrows():
        rows.append({"variable": c, "level": r[c], "n": int(r["count"]), "default_rate": round(r["mean"], 4)})
grp = pd.DataFrame(rows)
grp.to_csv(TABLES / "eda_default_rate_by_group.csv", index=False)

# Limit balance and utilisation by default status and sex.
bystatus = df.groupby([TARGET])[["LIMIT_BAL", "UTILISATION", "PAY_RATIO", "MAX_DELAY", "AGE"]].median().round(3)
bystatus.to_csv(TABLES / "eda_medians_by_default.csv")
bysex = df.groupby("SEX")[["LIMIT_BAL", "UTILISATION", "PAY_RATIO", "MAX_DELAY", "AGE"]].median().round(3)
bysex.to_csv(TABLES / "eda_medians_by_sex.csv")

# ---- Multivariate: correlation of numeric predictors with target ------------
corr_cols = ["LIMIT_BAL", "AGE"] + PAY_COLS + ["BILL_AMT1", "PAY_AMT1"] + ENGINEERED + [TARGET]
corr = df[corr_cols].corr(method="spearman").round(3)
corr.to_csv(TABLES / "eda_spearman_correlation.csv")

# ---- Figures ------------------------------------------------------------------
# Figure: default rate by sex and age band
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
sex_lab = {1: "Male", 2: "Female"}
g = df.groupby("SEX")[TARGET].mean()
axes[0].bar([sex_lab[i] for i in g.index], g.values, color=[DARK, GREY], width=0.6)
axes[0].set_title("Default rate by sex"); axes[0].set_ylim(0, 0.35)
for i, v in enumerate(g.values): axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center")
order = ["<30", "30-39", "40-49", ">=50"]
g = df.groupby("AGE_BAND")[TARGET].mean().reindex(order)
axes[1].bar(order, g.values, color=DARK, width=0.6); axes[1].set_title("Default rate by age band"); axes[1].set_ylim(0, 0.35)
for i, v in enumerate(g.values): axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center")
g = df.groupby("PAY_0")[TARGET].mean()
axes[2].bar(g.index.astype(str), g.values, color=ACCENT, width=0.7); axes[2].set_title("Default rate by September repayment code (PAY_0)")
axes[2].set_ylim(0, 1); axes[2].set_xlabel("PAY_0 code")
for ax in axes: ax.set_ylabel("Default rate")
plt.tight_layout(); plt.savefig(FIGURES / "eda_default_rate_by_group.png", bbox_inches="tight"); plt.close()

# Figure: distributions of credit limit and utilisation by default status
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
for y, col, lab in [(0, GREY, "No default"), (1, DARK, "Default")]:
    axes[0].hist(np.log10(df.loc[df[TARGET] == y, "LIMIT_BAL"]), bins=40, alpha=0.6, color=col, label=lab, density=True)
    u = df.loc[df[TARGET] == y, "UTILISATION"].clip(-0.5, 1.5)
    axes[1].hist(u, bins=40, alpha=0.6, color=col, label=lab, density=True)
axes[0].set_xlabel("log10 credit limit (NT$)"); axes[0].set_title("Credit limit by outcome"); axes[0].legend(frameon=False)
axes[1].set_xlabel("Utilisation (BILL_AMT1 / LIMIT_BAL, clipped to [-0.5, 1.5])"); axes[1].set_title("Utilisation by outcome")
for ax in axes: ax.set_ylabel("Density")
plt.tight_layout(); plt.savefig(FIGURES / "eda_distributions.png", bbox_inches="tight"); plt.close()

# Figure: Spearman correlation heatmap
fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=90)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        v = corr.values[i, j]
        if abs(v) >= 0.3 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman correlation")
ax.set_title("Spearman correlation among predictors and DEFAULT")
plt.tight_layout(); plt.savefig(FIGURES / "eda_correlation.png", bbox_inches="tight"); plt.close()

# Figure: credit limit by sex (the suspected proxy)
fig, ax = plt.subplots(figsize=(5.5, 2.8))
data = [np.log10(df.loc[df["SEX"] == s, "LIMIT_BAL"]) for s in (1, 2)]
ax.boxplot(data, tick_labels=["Male", "Female"], widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#e6e6e6", color=DARK), medianprops=dict(color=ACCENT))
ax.set_ylabel("log10 credit limit (NT$)"); ax.set_title("Credit limit by sex")
plt.tight_layout(); plt.savefig(FIGURES / "eda_limit_by_sex.png", bbox_inches="tight"); plt.close()

print(summary[["mean", "50%", "min", "max", "skew"]].head(8))
print(grp[grp.variable.isin(["SEX", "AGE_BAND"])])
print("median limit by sex:", bysex["LIMIT_BAL"].to_dict())
print("spearman with DEFAULT:", corr[TARGET].drop(TARGET).sort_values().round(3).to_dict())

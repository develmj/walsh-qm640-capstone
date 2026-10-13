"""RQ3: fairness of simulated decisions at a 75% retention cut-off. Cut-offs are
set per fold on the calibration partition and applied to the held-out fold.
Primary: unaware XGBoost. Comparators: aware XGBoost, unaware logistic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, PROCESSED, RETENTION_PRIMARY, TABLES, TARGET, load_clean
from evaluate import audit, bootstrap_audit, fdr_gap_test
from policy import apply_cutoff, cutoff_for_retention

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

df = load_clean().set_index("ID")
scores = pd.read_parquet(PROCESSED / "scores.parquet")


def decisions(fs, model, retention):
    """Return (y, sex, flag, ids) for pooled held-out decisions at `retention`."""
    s = scores[(scores.feature_set == fs) & (scores.model == model)]
    test = s[s.partition == "test"].set_index("ID")
    flag = np.zeros(len(test), dtype=int)
    thresholds = {}
    for k in range(5):
        cal = s[(s.partition == "cal") & (s.fold == k)]
        thr = cutoff_for_retention(cal["p_raw"].values, retention)
        thresholds[k] = thr
        fm = (test["fold"] == k).values
        flag[fm] = apply_cutoff(test["p_raw"].values[fm], thr)
    y = df.loc[test.index, TARGET].values
    sex = df.loc[test.index, "SEX"].values
    return y, sex, flag, test.index, thresholds


configs = [("unaware", "xgb", "XGBoost, unaware (primary)"),
           ("aware", "xgb", "XGBoost, aware"),
           ("unaware", "lr", "Logistic, unaware"),
           ("unaware", "rf", "Random forest, unaware")]
summary, detail, boots = [], [], []
for fs, m, label in configs:
    y, sex, flag, ids, thr = decisions(fs, m, RETENTION_PRIMARY)
    a = audit(y, sex, flag)
    b = bootstrap_audit(y, sex, flag)
    z, p = fdr_gap_test(y, sex, flag)
    ci = {r.measure: (r.ci95_low, r.ci95_high) for r in b.itertuples()}
    summary.append({
        "configuration": label,
        "overall_retain_rate": a["overall_retain_rate"],
        "male_retain_rate": a["male_retain_rate"], "female_retain_rate": a["female_retain_rate"],
        "DI_ratio": a["DI_ratio"], "DI_ci95_low": ci["DI_ratio"][0], "DI_ci95_high": ci["DI_ratio"][1],
        "male_FDR": a["male_FDR"], "female_FDR": a["female_FDR"],
        "FDR_gap": a["FDR_gap"], "FDR_gap_ci95_low": ci["FDR_gap"][0], "FDR_gap_ci95_high": ci["FDR_gap"][1],
        "FDR_gap_z": z, "FDR_gap_p": p,
        "male_TPR": a["male_TPR"], "female_TPR": a["female_TPR"],
        "TPR_gap": a["TPR_gap"], "TPR_gap_ci95_low": ci["TPR_gap"][0], "TPR_gap_ci95_high": ci["TPR_gap"][1],
        "EO_diff": a["EO_diff"],
        "male_default_rate_retained": a["male_default_rate_retained"], "female_default_rate_retained": a["female_default_rate_retained"],
        "overall_default_rate_retained": a["overall_default_rate_retained"],
        "benchmark_DI_ge_0.80": a["DI_ratio"] >= 0.80,
        "benchmark_FDR_gap_within_0.05": abs(a["FDR_gap"]) <= 0.05,
        "benchmark_FDR_gap_CI_within_0.05": (ci["FDR_gap"][0] >= -0.05) and (ci["FDR_gap"][1] <= 0.05),
    })
    for lab in ("male", "female"):
        detail.append({"configuration": label, "sex": lab, "n": a[f"{lab}_n"], "TP": a[f"{lab}_TP"], "FP": a[f"{lab}_FP"],
                       "FN": a[f"{lab}_FN"], "TN": a[f"{lab}_TN"], "retain_rate": a[f"{lab}_retain_rate"],
                       "FDR": a[f"{lab}_FDR"], "TPR": a[f"{lab}_TPR"], "precision": a[f"{lab}_precision"],
                       "default_rate_retained": a[f"{lab}_default_rate_retained"]})
    b["configuration"] = label
    boots.append(b)
sum_tab = pd.DataFrame(summary)
sum_tab["FDR_gap_p"] = sum_tab["FDR_gap_p"].map(lambda v: f"{v:.3g}")
sum_tab.to_csv(TABLES / "rq3_fairness_summary.csv", index=False, float_format="%.4f")
pd.DataFrame(detail).to_csv(TABLES / "rq3_confusion_by_sex.csv", index=False, float_format="%.4f")
pd.concat(boots).to_csv(TABLES / "rq3_bootstrap_intervals.csv", index=False, float_format="%.4f")

# ---- Figure: gaps with intervals for the four configurations -----------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
labels = [c[2] for c in configs]
ypos = np.arange(len(labels))[::-1]
for ax, key, title, bench in ((axes[0], "DI_ratio", "Disparate impact ratio (RETAIN)", 0.80),
                              (axes[1], "FDR_gap", "False-decline rate gap, male − female", 0.05),
                              (axes[2], "TPR_gap", "True-positive rate gap, male − female", None)):
    v = sum_tab[key].values
    pre = "DI" if key == "DI_ratio" else key
    lo = sum_tab[f"{pre}_ci95_low"].values; hi = sum_tab[f"{pre}_ci95_high"].values
    ax.errorbar(v, ypos, xerr=[v - lo, hi - v], fmt="o", color="#2b2b2b", ms=4, capsize=3, lw=1)
    if key == "DI_ratio":
        ax.axvline(bench, color="#1f5f8b", ls="--", lw=1); ax.set_xlim(0.7, 1.05)
    elif key == "FDR_gap":
        ax.axvline(-bench, color="#1f5f8b", ls="--", lw=1); ax.axvline(bench, color="#1f5f8b", ls="--", lw=1)
        ax.axvline(0, color="#cccccc", lw=0.8)
    else:
        ax.axvline(0, color="#cccccc", lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels))
    ax.set_title(title, fontsize=9)
plt.tight_layout(); plt.savefig(FIGURES / "rq3_fairness_gaps.png", bbox_inches="tight"); plt.close()

cols = ["configuration", "DI_ratio", "DI_ci95_low", "DI_ci95_high", "male_FDR", "female_FDR", "FDR_gap",
        "FDR_gap_ci95_low", "FDR_gap_ci95_high", "FDR_gap_p", "TPR_gap", "overall_default_rate_retained"]
print(sum_tab[cols].round(4).to_string(index=False))
print(pd.DataFrame(detail).round(4).to_string(index=False))

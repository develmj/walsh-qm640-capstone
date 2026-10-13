"""RQ1: model comparison on pooled held-out predictions. DeLong CIs for each
model, the pre-specified paired contrast XGBoost vs logistic, KS, Brier,
reliability, and ROC figure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import FIGURES, PROCESSED, RETENTION_PRIMARY, TABLES, TARGET, delong_paired, load_clean
from evaluate import performance, reliability
from policy import apply_cutoff, cutoff_for_retention

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

df = load_clean().set_index("ID")
scores = pd.read_parquet(PROCESSED / "scores.parquet")
MODEL_LABEL = {"lr": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost"}

rows, rel_rows = [], []
pooled = {}
for fs in ("unaware", "aware"):
    for m in ("lr", "rf", "xgb"):
        s = scores[(scores.feature_set == fs) & (scores.model == m)]
        test = s[s.partition == "test"].set_index("ID")
        y = df.loc[test.index, TARGET].values
        # cut-off at 75% retention set per fold on the calibration partition
        flag = np.zeros(len(test), dtype=int)
        for k in range(5):
            cal = s[(s.partition == "cal") & (s.fold == k)]
            thr = cutoff_for_retention(cal["p_raw"].values, RETENTION_PRIMARY)
            fm = (test["fold"] == k).values
            flag[fm] = apply_cutoff(test["p_raw"].values[fm], thr)
        perf = performance(y, test["p_raw"].values, test["p_cal"].values, flag)
        perf.update({"feature_set": fs, "model": MODEL_LABEL[m], "n": len(test), "retain_rate_at_cutoff": 1 - flag.mean()})
        rows.append(perf)
        pooled[(fs, m)] = (test["p_raw"].values, test["p_cal"].values, y, test.index)
        for kind, p in (("raw", test["p_raw"].values), ("calibrated", test["p_cal"].values)):
            r = reliability(p, y)
            r["feature_set"] = fs; r["model"] = MODEL_LABEL[m]; r["probabilities"] = kind
            rel_rows.append(r)
perf_tab = pd.DataFrame(rows)[["feature_set", "model", "n", "AUC", "AUC_se", "AUC_ci95_low", "AUC_ci95_high",
                               "KS", "Brier_raw", "Brier_calibrated", "F1_at_cutoff", "retain_rate_at_cutoff"]]
perf_tab.to_csv(TABLES / "rq1_performance.csv", index=False, float_format="%.4f")
pd.concat(rel_rows).to_csv(TABLES / "rq1_reliability.csv", index=False, float_format="%.4f")

# Pre-specified paired contrast: XGBoost vs logistic (unaware is primary); RF reported as secondary.
contrasts = []
for fs in ("unaware", "aware"):
    p_x, _, y, idx = pooled[(fs, "xgb")]
    p_l = pooled[(fs, "lr")][0]; p_r = pooled[(fs, "rf")][0]
    assert (pooled[(fs, "lr")][3] == idx).all() and (pooled[(fs, "rf")][3] == idx).all()
    for name, pa, pb in (("XGBoost vs logistic (pre-specified)", p_x, p_l),
                         ("Random forest vs logistic (secondary)", p_r, p_l),
                         ("XGBoost vs random forest (secondary)", p_x, p_r)):
        a, b, diff, se, z, p, corr = delong_paired(pa, pb, y)
        contrasts.append({"feature_set": fs, "contrast": name, "AUC_a": a, "AUC_b": b, "difference": diff,
                          "se_diff": se, "ci95_low": diff - 1.96 * se, "ci95_high": diff + 1.96 * se,
                          "z": z, "p_two_sided": p, "p_one_sided": p / 2 if diff > 0 else 1 - p / 2,
                          "paired_auc_correlation": corr})
con_tab = pd.DataFrame(contrasts)
con_tab.to_csv(TABLES / "rq1_delong_contrasts.csv", index=False, float_format="%.6g")

# Per-fold AUCs for stability
fold_rows = []
for fs in ("unaware", "aware"):
    for m in ("lr", "rf", "xgb"):
        s = scores[(scores.feature_set == fs) & (scores.model == m) & (scores.partition == "test")].set_index("ID")
        for k in range(5):
            sk = s[s.fold == k]
            from sklearn.metrics import roc_auc_score
            fold_rows.append({"feature_set": fs, "model": MODEL_LABEL[m], "fold": k,
                              "AUC": roc_auc_score(df.loc[sk.index, TARGET].values, sk["p_raw"].values)})
pd.DataFrame(fold_rows).pivot_table(index=["feature_set", "model"], columns="fold", values="AUC").round(4) \
    .to_csv(TABLES / "rq1_auc_by_fold.csv")

# ---- Figures: ROC (unaware) and reliability (unaware) ------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
colors = {"lr": "#9a9a9a", "rf": "#1f5f8b", "xgb": "#2b2b2b"}
for m in ("lr", "rf", "xgb"):
    p, pc, y, _ = pooled[("unaware", m)]
    fpr, tpr, _ = roc_curve(y, p)
    auc = perf_tab[(perf_tab.feature_set == "unaware") & (perf_tab.model == MODEL_LABEL[m])]["AUC"].iloc[0]
    axes[0].plot(fpr, tpr, color=colors[m], lw=1.4, label=f"{MODEL_LABEL[m]} (AUC {auc:.3f})")
    r = reliability(pc, y)
    axes[1].plot(r["mean_predicted"], r["observed_rate"], marker="o", ms=3, color=colors[m], lw=1.2, label=MODEL_LABEL[m])
axes[0].plot([0, 1], [0, 1], ls="--", color="#cccccc", lw=1)
axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
axes[0].set_title("ROC curves, unaware feature set, pooled held-out folds"); axes[0].legend(frameon=False, loc="lower right")
axes[1].plot([0, 1], [0, 1], ls="--", color="#cccccc", lw=1)
axes[1].set_xlabel("Mean predicted probability (isotonic-calibrated)"); axes[1].set_ylabel("Observed default rate")
axes[1].set_title("Reliability by decile, unaware feature set"); axes[1].legend(frameon=False, loc="upper left")
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(FIGURES / "rq1_roc_reliability.png", bbox_inches="tight"); plt.close()

print(perf_tab.round(4).to_string(index=False))
print(con_tab[["feature_set", "contrast", "AUC_a", "AUC_b", "difference", "ci95_low", "ci95_high", "z", "p_two_sided"]].round(4).to_string(index=False))

"""RQ4: mitigation trade-offs. (a) Reweighing: retrain unaware XGBoost with
Kamiran-Calders weights computed on the fit partition, same hyperparameters as
the tuned model for that fold, isotonic recalibration on the calibration
partition, held-out scoring, paired-AUC non-inferiority test. (b) Threshold
adjustment: per-sex thresholds found on the calibration partition that
equalise the false-decline rate at a common retention rate. Trade-off table
across retention rates for the unmitigated and both mitigated configurations."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FEATURES_UNAWARE, FIGURES, PROCESSED, RETENTION_GRID, RETENTION_PRIMARY, TABLES, TARGET,
                    delong_ci, load_clean, load_folds)
from evaluate import audit, bootstrap_audit, fdr_gap_test, noninferiority
from policy import (apply_cutoff, apply_group_thresholds, cutoff_for_retention, group_thresholds,
                    reweighing_table, reweighing_weights)
from train import run

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

df = load_clean()
folds = load_folds().set_index("ID").loc[df["ID"]].reset_index()
scores = pd.read_parquet(PROCESSED / "scores.parquet")
dfi = df.set_index("ID")

# ---------------------------------------------------------------------------
# (a) Reweighing: retrain unaware XGBoost with weights from the fit partition.
# ---------------------------------------------------------------------------
best = pd.read_json(PROCESSED / "xgb_best_params.json")
best_u = best[best.feature_set == "unaware"].set_index("fold")
param_cols = [c for c in best_u.columns if c not in ("feature_set", "inner_cv_auc")]
params_by_fold = {k: {c: (int(v) if c in ("n_estimators", "max_depth", "min_child_weight") else float(v))
                      for c, v in best_u.loc[k, param_cols].items()} for k in range(5)}

# Weights are computed on each fold's fit partition only. train.run() takes a
# single weight vector aligned with df, so build one vector per fold and call
# run() fold by fold via a weight matrix.
w_all = np.ones(len(df))
rw_tables = []
for k in range(5):
    fit_mask = folds[f"part_fold{k}"].values == "fit"
    w_fit = reweighing_weights(df.loc[fit_mask, "SEX"].values, df.loc[fit_mask, TARGET].values)
    w_all_k = np.ones(len(df)); w_all_k[fit_mask] = w_fit
    t = reweighing_table(df.loc[fit_mask, "SEX"].values, df.loc[fit_mask, TARGET].values); t["fold"] = k
    rw_tables.append(t)
    if k == 0:
        w_all = w_all_k
pd.concat(rw_tables).to_csv(TABLES / "rq4_reweighing_weights.csv", index=False)

# Because weights differ by fold only on the fit rows of that fold, run() can be
# called once per fold with that fold's weight vector by filtering folds.
rw_scores = []
for k in range(5):
    fit_mask = folds[f"part_fold{k}"].values == "fit"
    w_k = np.ones(len(df)); w_k[fit_mask] = reweighing_weights(df.loc[fit_mask, "SEX"].values, df.loc[fit_mask, TARGET].values)
    folds_k = folds.copy()
    for j in range(5):
        if j != k:
            folds_k[f"part_fold{j}"] = "skip"
    s, _ = run("unaware", FEATURES_UNAWARE, df, folds_k, sample_weight=w_k,
               xgb_params_override=params_by_fold, models=("xgb",), tag="unaware_reweighed")
    rw_scores.append(s[s.fold == k])
rw = pd.concat(rw_scores, ignore_index=True)
rw.to_parquet(PROCESSED / "scores_reweighed.parquet", index=False)

base = scores[(scores.feature_set == "unaware") & (scores.model == "xgb")]
base_test = base[base.partition == "test"].set_index("ID")
rw_test = rw[rw.partition == "test"].set_index("ID").loc[base_test.index]
y = dfi.loc[base_test.index, TARGET].values
sex = dfi.loc[base_test.index, "SEX"].values

ni = noninferiority(rw_test["p_raw"].values, base_test["p_raw"].values, y, margin=0.01)
auc_rw = delong_ci(rw_test["p_raw"].values, y)
ni.update({"AUC_reweighed_ci95_low": auc_rw[2], "AUC_reweighed_ci95_high": auc_rw[3]})
pd.DataFrame([ni]).to_csv(TABLES / "rq4_reweighing_noninferiority.csv", index=False, float_format="%.6g")


# ---------------------------------------------------------------------------
# Decision helpers for the three configurations at any retention rate.
# ---------------------------------------------------------------------------
def flags_single(s_all, retention):
    test = s_all[s_all.partition == "test"].set_index("ID").loc[base_test.index]
    flag = np.zeros(len(test), dtype=int)
    for k in range(5):
        cal = s_all[(s_all.partition == "cal") & (s_all.fold == k)]
        thr = cutoff_for_retention(cal["p_raw"].values, retention)
        fm = (test["fold"] == k).values
        flag[fm] = apply_cutoff(test["p_raw"].values[fm], thr)
    return flag


def flags_group(s_all, retention):
    test = s_all[s_all.partition == "test"].set_index("ID").loc[base_test.index]
    flag = np.zeros(len(test), dtype=int)
    thr_log = []
    for k in range(5):
        cal = s_all[(s_all.partition == "cal") & (s_all.fold == k)].set_index("ID")
        thr, diag = group_thresholds(cal["p_raw"].values, dfi.loc[cal.index, "SEX"].values,
                                     dfi.loc[cal.index, TARGET].values, retention)
        thr_log.append({"retention": retention, "fold": k, "threshold_male": thr[1], "threshold_female": thr[2], **diag})
        fm = (test["fold"] == k).values
        flag[fm] = apply_group_thresholds(test["p_raw"].values[fm], sex[fm], thr)
    return flag, thr_log


configs = {"Unmitigated (single threshold)": lambda r: (flags_single(base, r), None),
           "Reweighing (single threshold)": lambda r: (flags_single(rw, r), None),
           "Threshold adjustment (per-sex thresholds)": lambda r: flags_group(base, r)}
auc_by_config = {"Unmitigated (single threshold)": ni["AUC_base"], "Reweighing (single threshold)": ni["AUC_new"],
                 "Threshold adjustment (per-sex thresholds)": ni["AUC_base"]}

rows, thr_rows, boot_rows = [], [], []
for r in RETENTION_GRID:
    for name, fn in configs.items():
        flag, thr_log = fn(r)
        if thr_log:
            thr_rows.extend(thr_log)
        a = audit(y, sex, flag)
        b = bootstrap_audit(y, sex, flag)
        ci = {x.measure: (x.ci95_low, x.ci95_high) for x in b.itertuples()}
        z, p = fdr_gap_test(y, sex, flag)
        rows.append({"retention_target": r, "configuration": name,
                     "overall_retain_rate": a["overall_retain_rate"],
                     "default_rate_retained": a["overall_default_rate_retained"],
                     "default_rate_retained_ci95_low": ci["overall_default_rate_retained"][0],
                     "default_rate_retained_ci95_high": ci["overall_default_rate_retained"][1],
                     "AUC": auc_by_config[name],
                     "male_retain_rate": a["male_retain_rate"], "female_retain_rate": a["female_retain_rate"],
                     "DI_ratio": a["DI_ratio"], "DI_ci95_low": ci["DI_ratio"][0], "DI_ci95_high": ci["DI_ratio"][1],
                     "male_FDR": a["male_FDR"], "female_FDR": a["female_FDR"],
                     "FDR_gap": a["FDR_gap"], "FDR_gap_ci95_low": ci["FDR_gap"][0], "FDR_gap_ci95_high": ci["FDR_gap"][1],
                     "FDR_gap_p": p, "TPR_gap": a["TPR_gap"], "TPR_gap_ci95_low": ci["TPR_gap"][0],
                     "TPR_gap_ci95_high": ci["TPR_gap"][1],
                     "overall_TPR": a["overall_TPR"],
                     "meets_DI_0.80": a["DI_ratio"] >= 0.80, "meets_FDR_gap_0.05": abs(a["FDR_gap"]) <= 0.05})
trade = pd.DataFrame(rows)
trade["FDR_gap_p"] = trade["FDR_gap_p"].map(lambda v: f"{v:.3g}")
trade.to_csv(TABLES / "rq4_tradeoff_table.csv", index=False, float_format="%.4f")
pd.DataFrame(thr_rows).to_csv(TABLES / "rq4_group_thresholds.csv", index=False, float_format="%.5f")

# Primary comparison at 75% retention
prim = trade[trade.retention_target == RETENTION_PRIMARY].copy()
base_row = prim[prim.configuration.str.startswith("Unmitigated")].iloc[0]
prim["delta_default_rate_retained_vs_unmitigated"] = prim["default_rate_retained"] - base_row["default_rate_retained"]
prim["delta_AUC_vs_unmitigated"] = prim["AUC"] - base_row["AUC"]
prim.to_csv(TABLES / "rq4_primary_75pct.csv", index=False, float_format="%.4f")

# ---- Figure: fairness gap and bad rate across retention rates ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
styles = {"Unmitigated (single threshold)": ("#2b2b2b", "o", "-"),
          "Reweighing (single threshold)": ("#1f5f8b", "s", "-"),
          "Threshold adjustment (per-sex thresholds)": ("#9a9a9a", "^", "--")}
for name, (col, mk, ls) in styles.items():
    t = trade[trade.configuration == name]
    x = t["overall_retain_rate"].values
    axes[0].errorbar(x, t["FDR_gap"], yerr=[t["FDR_gap"] - t["FDR_gap_ci95_low"], t["FDR_gap_ci95_high"] - t["FDR_gap"]],
                     color=col, marker=mk, ls=ls, ms=4, lw=1.2, capsize=2, label=name)
    axes[1].plot(x, t["DI_ratio"], color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=name)
    axes[2].plot(x, t["default_rate_retained"], color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=name)
axes[0].axhline(0.05, color="#bbbbbb", ls=":", lw=1); axes[0].axhline(-0.05, color="#bbbbbb", ls=":", lw=1); axes[0].axhline(0, color="#dddddd", lw=0.8)
axes[0].set_title("False-decline rate gap, male − female"); axes[0].set_ylabel("Gap (95% bootstrap CI)")
axes[1].axhline(0.80, color="#bbbbbb", ls=":", lw=1); axes[1].set_title("Disparate impact ratio on RETAIN"); axes[1].set_ylabel("DI ratio")
axes[2].set_title("Default rate among retained cardholders"); axes[2].set_ylabel("Default rate")
for ax in axes:
    ax.set_xlabel("Overall retention rate"); ax.set_xticks(RETENTION_GRID)
axes[2].legend(frameon=False, fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(FIGURES / "rq4_tradeoff.png", bbox_inches="tight"); plt.close()

print(pd.DataFrame([ni]).T)
cols = ["retention_target", "configuration", "overall_retain_rate", "default_rate_retained", "AUC", "DI_ratio",
        "FDR_gap", "FDR_gap_ci95_low", "FDR_gap_ci95_high", "TPR_gap", "meets_DI_0.80", "meets_FDR_gap_0.05"]
print(trade[cols].round(4).to_string(index=False))

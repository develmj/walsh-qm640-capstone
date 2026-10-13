"""Sensitivity runs on the coding assumptions and a SHAP decomposition by sex.

Sensitivity A (unaware XGBoost): collapse the undocumented repayment codes -2
and 0 into a single non-delinquent code 0 and recompute MAX_DELAY.
Sensitivity B (aware XGBoost): merge the undocumented EDUCATION level 0 into
category 4 (other) and MARRIAGE level 0 into category 3 (other).
Both reuse the tuned hyperparameters per fold and report AUC and the RQ3
measures at 75% retention against the primary runs.

SHAP: refit the fold-0 unaware XGBoost with its tuned parameters and compute
TreeExplainer attributions on the fold-0 held-out rows, summarised by sex."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import (FEATURES_AWARE, FEATURES_UNAWARE, FIGURES, PAY_COLS, PROCESSED, RETENTION_PRIMARY, SEED, TABLES,
                    TARGET, delong_ci, load_clean, load_folds)
from evaluate import audit, bootstrap_audit
from policy import apply_cutoff, cutoff_for_retention
from train import make_xgb, run

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

df = load_clean()
folds = load_folds().set_index("ID").loc[df["ID"]].reset_index()
dfi = df.set_index("ID")
scores = pd.read_parquet(PROCESSED / "scores.parquet")
best = pd.read_json(PROCESSED / "xgb_best_params.json")
param_cols = [c for c in best.columns if c not in ("feature_set", "fold", "inner_cv_auc")]


def params_for(fs):
    b = best[best.feature_set == fs].set_index("fold")
    return {k: {c: (int(v) if c in ("n_estimators", "max_depth", "min_child_weight") else float(v))
                for c, v in b.loc[k, param_cols].items()} for k in range(5)}


def summarise(s_all, label):
    test = s_all[s_all.partition == "test"].set_index("ID")
    y = dfi.loc[test.index, TARGET].values; sex = dfi.loc[test.index, "SEX"].values
    flag = np.zeros(len(test), dtype=int)
    for k in range(5):
        cal = s_all[(s_all.partition == "cal") & (s_all.fold == k)]
        thr = cutoff_for_retention(cal["p_raw"].values, RETENTION_PRIMARY)
        fm = (test["fold"] == k).values
        flag[fm] = apply_cutoff(test["p_raw"].values[fm], thr)
    auc, se, lo, hi = delong_ci(test["p_raw"].values, y)
    a = audit(y, sex, flag)
    b = bootstrap_audit(y, sex, flag)
    ci = {r.measure: (r.ci95_low, r.ci95_high) for r in b.itertuples()}
    return {"run": label, "AUC": auc, "AUC_ci95_low": lo, "AUC_ci95_high": hi, "DI_ratio": a["DI_ratio"],
            "FDR_gap": a["FDR_gap"], "FDR_gap_ci95_low": ci["FDR_gap"][0], "FDR_gap_ci95_high": ci["FDR_gap"][1],
            "TPR_gap": a["TPR_gap"], "default_rate_retained": a["overall_default_rate_retained"]}


rows = []
rows.append(summarise(scores[(scores.feature_set == "unaware") & (scores.model == "xgb")], "Primary: unaware XGBoost"))
# Sensitivity A
dfA = df.copy()
for c in PAY_COLS:
    dfA[c] = dfA[c].replace({-2: 0})
dfA["MAX_DELAY"] = dfA[PAY_COLS].max(axis=1)
sA, _ = run("unaware", FEATURES_UNAWARE, dfA, folds, xgb_params_override=params_for("unaware"), models=("xgb",), tag="sensA")
rows.append(summarise(sA, "Sensitivity A: PAY codes -2 and 0 collapsed (unaware XGBoost)"))
# Sensitivity B
rows.append(summarise(scores[(scores.feature_set == "aware") & (scores.model == "xgb")], "Primary: aware XGBoost"))
dfB = df.copy()
dfB["EDUCATION"] = dfB["EDUCATION"].replace({0: 4})
dfB["MARRIAGE"] = dfB["MARRIAGE"].replace({0: 3})
sB, _ = run("aware", FEATURES_AWARE, dfB, folds, xgb_params_override=params_for("aware"), models=("xgb",), tag="sensB")
rows.append(summarise(sB, "Sensitivity B: undocumented EDUCATION and MARRIAGE merged into other (aware XGBoost)"))
sens = pd.DataFrame(rows)
sens.to_csv(TABLES / "rq_sensitivity_coding.csv", index=False, float_format="%.4f")
print(sens.round(4).to_string(index=False))

# ---- SHAP on fold 0, unaware XGBoost ----------------------------------------
k = 0
part = folds[f"part_fold{k}"].values
fit_idx = np.where(part == "fit")[0]; test_idx = np.where(part == "test")[0]
X = df[FEATURES_UNAWARE]; y = df[TARGET].values
spw = (y[fit_idx] == 0).sum() / (y[fit_idx] == 1).sum()
model = make_xgb(spw).set_params(**params_for("unaware")[k]).fit(X.iloc[fit_idx], y[fit_idx])
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X.iloc[test_idx])
sex_test = df["SEX"].values[test_idx]
abs_m = np.abs(sv[sex_test == 1]).mean(axis=0); abs_f = np.abs(sv[sex_test == 2]).mean(axis=0)
mean_m = sv[sex_test == 1].mean(axis=0); mean_f = sv[sex_test == 2].mean(axis=0)
shap_tab = pd.DataFrame({"feature": FEATURES_UNAWARE, "mean_abs_shap_all": np.abs(sv).mean(axis=0),
                         "mean_abs_shap_male": abs_m, "mean_abs_shap_female": abs_f,
                         "mean_shap_male": mean_m, "mean_shap_female": mean_f,
                         "mean_shap_difference_male_minus_female": mean_m - mean_f})
shap_tab = shap_tab.sort_values("mean_abs_shap_all", ascending=False)
shap_tab.to_csv(TABLES / "rq3_shap_by_sex.csv", index=False, float_format="%.5f")
print(shap_tab.head(12).round(4).to_string(index=False))

top = shap_tab.head(12)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ypos = np.arange(len(top))[::-1]
axes[0].barh(ypos + 0.18, top["mean_abs_shap_male"], height=0.36, color="#2b2b2b", label="Male")
axes[0].barh(ypos - 0.18, top["mean_abs_shap_female"], height=0.36, color="#9a9a9a", label="Female")
axes[0].set_yticks(ypos); axes[0].set_yticklabels(top["feature"]); axes[0].set_xlabel("Mean |SHAP| (log-odds)")
axes[0].set_title("Feature influence by sex, fold 0 held-out rows"); axes[0].legend(frameon=False)
diff = top["mean_shap_difference_male_minus_female"]
axes[1].barh(ypos, diff, color=["#1f5f8b" if d > 0 else "#9a9a9a" for d in diff], height=0.6)
axes[1].axvline(0, color="#cccccc", lw=0.8)
axes[1].set_yticks(ypos); axes[1].set_yticklabels([""] * len(top)); axes[1].set_xlabel("Mean SHAP, male − female (log-odds)")
axes[1].set_title("Direction of the group difference (positive raises male risk)")
plt.tight_layout(); plt.savefig(FIGURES / "rq3_shap_by_sex.png", bbox_inches="tight"); plt.close()

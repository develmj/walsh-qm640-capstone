"""RQ4 extensions and an RQ3 robustness check.

(a) Fairlearn ThresholdOptimizer cross-check on the direct-search per-sex
    thresholds: fitted on each fold's calibration partition with a
    false-positive-rate parity constraint, applied to the held-out fold.
    Its retention is not pinned to 75%, so achieved retention is reported.
(b) Reweighing applied to the AWARE XGBoost model, where the model can see
    sex and the weights can act directly, with the paired-AUC non-inferiority
    test and the fairness audit at 75% retention.
(c) RQ3 robustness: the primary unaware audit repeated by age band."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

from common import (FEATURES_AWARE, PROCESSED, RETENTION_PRIMARY, SEED, TABLES, TARGET, delong_ci, load_clean,
                    load_folds)
from evaluate import audit, bootstrap_audit, fdr_gap_test, noninferiority
from policy import apply_cutoff, cutoff_for_retention, reweighing_weights
from train import run

warnings.filterwarnings("ignore")
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


class Scorepassthrough(BaseEstimator, ClassifierMixin):
    """Exposes a precomputed score column as predict_proba for ThresholdOptimizer."""
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        p = np.asarray(X).ravel(); return np.column_stack([1 - p, p])
    def predict(self, X):
        return (np.asarray(X).ravel() > 0.5).astype(int)


base = scores[(scores.feature_set == "unaware") & (scores.model == "xgb")]
base_test = base[base.partition == "test"].set_index("ID")
y = dfi.loc[base_test.index, TARGET].values
sex = dfi.loc[base_test.index, "SEX"].values

# ---------------------------------------------------------------------------
# (a) ThresholdOptimizer cross-check
# ---------------------------------------------------------------------------
rows = []
flag_to = np.zeros(len(base_test), dtype=int)
for k in range(5):
    cal = base[(base.partition == "cal") & (base.fold == k)].set_index("ID")
    to = ThresholdOptimizer(estimator=Scorepassthrough().fit(None),
                            constraints="false_positive_rate_parity", objective="balanced_accuracy_score",
                            prefit=True, predict_method="predict_proba")
    to.fit(cal[["p_raw"]].values, dfi.loc[cal.index, TARGET].values, sensitive_features=dfi.loc[cal.index, "SEX"].values)
    fm = (base_test["fold"] == k).values
    flag_to[fm] = to.predict(base_test.loc[fm, ["p_raw"]].values, sensitive_features=sex[fm], random_state=SEED + k)
a = audit(y, sex, flag_to); b = bootstrap_audit(y, sex, flag_to)
ci = {r.measure: (r.ci95_low, r.ci95_high) for r in b.itertuples()}
z, p = fdr_gap_test(y, sex, flag_to)
rows.append({"configuration": "Fairlearn ThresholdOptimizer, FPR parity, balanced-accuracy objective (cross-check)",
             "overall_retain_rate": a["overall_retain_rate"], "default_rate_retained": a["overall_default_rate_retained"],
             "DI_ratio": a["DI_ratio"], "male_FDR": a["male_FDR"], "female_FDR": a["female_FDR"],
             "FDR_gap": a["FDR_gap"], "FDR_gap_ci95_low": ci["FDR_gap"][0], "FDR_gap_ci95_high": ci["FDR_gap"][1],
             "FDR_gap_p": f"{p:.3g}", "TPR_gap": a["TPR_gap"], "note": "retention chosen by the optimizer, not pinned to 75%; decisions may be randomised"})
# direct-search result at the same retention as the optimizer, for a like-for-like comparison
r_to = a["overall_retain_rate"]
from policy import apply_group_thresholds, group_thresholds
flag_ds = np.zeros(len(base_test), dtype=int)
for k in range(5):
    cal = base[(base.partition == "cal") & (base.fold == k)].set_index("ID")
    thr, _ = group_thresholds(cal["p_raw"].values, dfi.loc[cal.index, "SEX"].values, dfi.loc[cal.index, TARGET].values, r_to)
    fm = (base_test["fold"] == k).values
    flag_ds[fm] = apply_group_thresholds(base_test["p_raw"].values[fm], sex[fm], thr)
a2 = audit(y, sex, flag_ds); b2 = bootstrap_audit(y, sex, flag_ds); ci2 = {r.measure: (r.ci95_low, r.ci95_high) for r in b2.itertuples()}
z2, p2 = fdr_gap_test(y, sex, flag_ds)
rows.append({"configuration": f"Direct-search per-sex thresholds at the optimizer's retention ({r_to:.3f})",
             "overall_retain_rate": a2["overall_retain_rate"], "default_rate_retained": a2["overall_default_rate_retained"],
             "DI_ratio": a2["DI_ratio"], "male_FDR": a2["male_FDR"], "female_FDR": a2["female_FDR"],
             "FDR_gap": a2["FDR_gap"], "FDR_gap_ci95_low": ci2["FDR_gap"][0], "FDR_gap_ci95_high": ci2["FDR_gap"][1],
             "FDR_gap_p": f"{p2:.3g}", "TPR_gap": a2["TPR_gap"], "note": "deterministic"})
pd.DataFrame(rows).to_csv(TABLES / "rq4_thresholdoptimizer_crosscheck.csv", index=False, float_format="%.4f")
print(pd.DataFrame(rows)[["configuration", "overall_retain_rate", "default_rate_retained", "DI_ratio", "FDR_gap", "FDR_gap_ci95_low", "FDR_gap_ci95_high"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# (b) Reweighing on the aware model
# ---------------------------------------------------------------------------
aware = scores[(scores.feature_set == "aware") & (scores.model == "xgb")]
aware_test = aware[aware.partition == "test"].set_index("ID").loc[base_test.index]
rw_parts = []
for k in range(5):
    fit_mask = folds[f"part_fold{k}"].values == "fit"
    w_k = np.ones(len(df)); w_k[fit_mask] = reweighing_weights(df.loc[fit_mask, "SEX"].values, df.loc[fit_mask, TARGET].values)
    folds_k = folds.copy()
    for j in range(5):
        if j != k:
            folds_k[f"part_fold{j}"] = "skip"
    s, _ = run("aware", FEATURES_AWARE, df, folds_k, sample_weight=w_k, xgb_params_override=params_for("aware"),
               models=("xgb",), tag="aware_reweighed")
    rw_parts.append(s[s.fold == k])
rw_aware = pd.concat(rw_parts, ignore_index=True)
rw_aware.to_parquet(PROCESSED / "scores_reweighed_aware.parquet", index=False)
rw_aware_test = rw_aware[rw_aware.partition == "test"].set_index("ID").loc[base_test.index]
ni = noninferiority(rw_aware_test["p_raw"].values, aware_test["p_raw"].values, y, margin=0.01)


def flags_single(s_all, retention):
    test = s_all[s_all.partition == "test"].set_index("ID").loc[base_test.index]
    flag = np.zeros(len(test), dtype=int)
    for k in range(5):
        cal = s_all[(s_all.partition == "cal") & (s_all.fold == k)]
        thr = cutoff_for_retention(cal["p_raw"].values, retention)
        fm = (test["fold"] == k).values
        flag[fm] = apply_cutoff(test["p_raw"].values[fm], thr)
    return flag


rows_b = []
for label, s_all, auc in (("Aware XGBoost, unmitigated", aware, ni["AUC_base"]),
                          ("Aware XGBoost, reweighed", rw_aware, ni["AUC_new"])):
    flag = flags_single(s_all, RETENTION_PRIMARY)
    a = audit(y, sex, flag); b = bootstrap_audit(y, sex, flag); ci = {r.measure: (r.ci95_low, r.ci95_high) for r in b.itertuples()}
    z, p = fdr_gap_test(y, sex, flag)
    rows_b.append({"configuration": label, "AUC": auc, "overall_retain_rate": a["overall_retain_rate"],
                   "default_rate_retained": a["overall_default_rate_retained"], "DI_ratio": a["DI_ratio"],
                   "male_FDR": a["male_FDR"], "female_FDR": a["female_FDR"], "FDR_gap": a["FDR_gap"],
                   "FDR_gap_ci95_low": ci["FDR_gap"][0], "FDR_gap_ci95_high": ci["FDR_gap"][1], "FDR_gap_p": f"{p:.3g}",
                   "TPR_gap": a["TPR_gap"]})
tab_b = pd.DataFrame(rows_b)
tab_b.to_csv(TABLES / "rq4_reweighing_aware_model.csv", index=False, float_format="%.4f")
pd.DataFrame([ni]).to_csv(TABLES / "rq4_reweighing_aware_noninferiority.csv", index=False, float_format="%.6g")
print(tab_b.round(4).to_string(index=False))
print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ni.items()})

# ---------------------------------------------------------------------------
# (c) RQ3 robustness by age band (unaware XGBoost, 75% retention)
# ---------------------------------------------------------------------------
flag = flags_single(base, RETENTION_PRIMARY)
band = dfi.loc[base_test.index, "AGE_BAND"].values
rows_c = []
for bnd in ["<30", "30-39", "40-49", ">=50"]:
    m = band == bnd
    yb, fb = y[m], flag[m]
    nd = yb == 0
    fdr = fb[nd].mean(); tpr = fb[yb == 1].mean(); ret = 1 - fb.mean()
    se = np.sqrt(fdr * (1 - fdr) / nd.sum())
    rows_c.append({"age_band": bnd, "n": int(m.sum()), "retain_rate": ret, "FDR": fdr, "FDR_ci95_low": fdr - 1.96 * se,
                   "FDR_ci95_high": fdr + 1.96 * se, "TPR": tpr, "default_rate_retained": yb[fb == 0].mean()})
tab_c = pd.DataFrame(rows_c)
tab_c["DI_ratio_vs_max_band"] = tab_c["retain_rate"] / tab_c["retain_rate"].max()
tab_c.to_csv(TABLES / "rq3_age_band_audit.csv", index=False, float_format="%.4f")
print(tab_c.round(4).to_string(index=False))

"""Evaluation: performance metrics, fairness audit with stratified bootstrap
intervals, and non-inferiority test for paired AUCs."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

from common import N_BOOT, SEED, delong_ci, delong_paired


def audit(y, sex, flag) -> dict:
    """Per-sex confusion matrices and fairness measures. Positive prediction is
    FLAG = 1; the favourable decision is RETAIN = 1 - FLAG."""
    y = np.asarray(y).astype(int); sex = np.asarray(sex); flag = np.asarray(flag).astype(int)
    out = {}
    for s, lab in ((1, "male"), (2, "female")):
        m = sex == s
        tp = int(((flag == 1) & (y == 1) & m).sum()); fp = int(((flag == 1) & (y == 0) & m).sum())
        fn = int(((flag == 0) & (y == 1) & m).sum()); tn = int(((flag == 0) & (y == 0) & m).sum())
        n = tp + fp + fn + tn
        out[f"{lab}_n"] = n; out[f"{lab}_TP"] = tp; out[f"{lab}_FP"] = fp; out[f"{lab}_FN"] = fn; out[f"{lab}_TN"] = tn
        out[f"{lab}_retain_rate"] = (tn + fn) / n
        out[f"{lab}_FDR"] = fp / (fp + tn)          # false-decline rate among non-defaulters
        out[f"{lab}_TPR"] = tp / (tp + fn)          # share of defaulters flagged
        out[f"{lab}_default_rate_retained"] = fn / (tn + fn)
        out[f"{lab}_precision"] = tp / (tp + fp) if tp + fp else np.nan
    rr = np.array([out["male_retain_rate"], out["female_retain_rate"]])
    out["DI_ratio"] = rr.min() / rr.max()
    out["FDR_gap"] = out["male_FDR"] - out["female_FDR"]
    out["TPR_gap"] = out["male_TPR"] - out["female_TPR"]
    out["EO_diff"] = max(abs(out["FDR_gap"]), abs(out["TPR_gap"]))
    out["overall_retain_rate"] = (flag == 0).mean()
    out["overall_default_rate_retained"] = y[flag == 0].mean()
    out["overall_FDR"] = ((flag == 1) & (y == 0)).sum() / (y == 0).sum()
    out["overall_TPR"] = ((flag == 1) & (y == 1)).sum() / (y == 1).sum()
    return out


def bootstrap_audit(y, sex, flag, n_boot=N_BOOT, seed=SEED) -> pd.DataFrame:
    """Stratified bootstrap of the audit measures. Within a sex-by-outcome
    stratum the only random quantity is the count flagged, so the stratified
    resample is exactly a binomial draw per stratum."""
    y = np.asarray(y).astype(int); sex = np.asarray(sex); flag = np.asarray(flag).astype(int)
    rng = np.random.default_rng(seed)
    cells = {}
    for s in (1, 2):
        for yy in (0, 1):
            m = (sex == s) & (y == yy)
            n = int(m.sum()); p = flag[m].mean()
            cells[(s, yy)] = (n, rng.binomial(n, p, size=n_boot))
    res = {}
    for s, lab in ((1, "male"), (2, "female")):
        n0, fp = cells[(s, 0)]; n1, tp = cells[(s, 1)]
        tn, fn = n0 - fp, n1 - tp
        res[f"{lab}_retain_rate"] = (tn + fn) / (n0 + n1)
        res[f"{lab}_FDR"] = fp / n0
        res[f"{lab}_TPR"] = tp / n1
        res[f"{lab}_default_rate_retained"] = fn / (tn + fn)
    rr = np.vstack([res["male_retain_rate"], res["female_retain_rate"]])
    res["DI_ratio"] = rr.min(axis=0) / rr.max(axis=0)
    res["FDR_gap"] = res["male_FDR"] - res["female_FDR"]
    res["TPR_gap"] = res["male_TPR"] - res["female_TPR"]
    n0 = cells[(1, 0)][0] + cells[(2, 0)][0]; n1 = cells[(1, 1)][0] + cells[(2, 1)][0]
    fp_all = cells[(1, 0)][1] + cells[(2, 0)][1]; tp_all = cells[(1, 1)][1] + cells[(2, 1)][1]
    res["overall_default_rate_retained"] = (n1 - tp_all) / ((n0 - fp_all) + (n1 - tp_all))
    rows = []
    for k, v in res.items():
        rows.append({"measure": k, "boot_mean": v.mean(), "ci95_low": np.quantile(v, 0.025), "ci95_high": np.quantile(v, 0.975)})
    return pd.DataFrame(rows)


def fdr_gap_test(y, sex, flag):
    """Two-proportion z test on the false-decline rate, male vs female."""
    y = np.asarray(y).astype(int); sex = np.asarray(sex); flag = np.asarray(flag).astype(int)
    nd = y == 0
    cnt = [int((flag[nd & (sex == 1)] == 1).sum()), int((flag[nd & (sex == 2)] == 1).sum())]
    nob = [int((nd & (sex == 1)).sum()), int((nd & (sex == 2)).sum())]
    z, p = proportions_ztest(cnt, nob)
    return float(z), float(p)


def performance(y, p_raw, p_cal, flag=None) -> dict:
    from sklearn.metrics import f1_score
    from common import brier, ks_statistic
    auc, se, lo, hi = delong_ci(p_raw, y)
    out = {"AUC": auc, "AUC_se": se, "AUC_ci95_low": lo, "AUC_ci95_high": hi,
           "KS": ks_statistic(p_raw, y), "Brier_raw": brier(p_raw, y), "Brier_calibrated": brier(p_cal, y)}
    if flag is not None:
        out["F1_at_cutoff"] = f1_score(y, flag)
    return out


def noninferiority(scores_new, scores_base, y, margin=0.01, alpha=0.05) -> dict:
    """One-sided non-inferiority of AUC_new against AUC_base - margin using the
    paired DeLong variance. Non-inferior if the lower one-sided (1 - alpha)
    bound of the difference exceeds -margin."""
    auc_new, auc_base, diff, se, z, p2, corr = delong_paired(scores_new, scores_base, y)
    lower = diff - norm.ppf(1 - alpha) * se
    z_ni = (diff + margin) / se
    p_ni = norm.sf(z_ni)  # H0: diff <= -margin
    return {"AUC_new": auc_new, "AUC_base": auc_base, "difference": diff, "se_diff": se,
            "lower_bound_one_sided_95": lower, "margin": -margin, "z_noninferiority": z_ni,
            "p_noninferiority": p_ni, "non_inferior": bool(lower > -margin), "paired_auc_correlation": corr}


def reliability(p, y, n_bins=10) -> pd.DataFrame:
    p = np.asarray(p); y = np.asarray(y)
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9
    idx = np.digitize(p, bins[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b + 1, "n": int(m.sum()), "mean_predicted": p[m].mean(), "observed_rate": y[m].mean()})
    return pd.DataFrame(rows)

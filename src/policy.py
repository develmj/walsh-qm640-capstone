"""Policy calibration: retention cut-offs, group-specific thresholds, and
reweighing weights. Every function here is fitted on the calibration partition
(or the fit partition for reweighing) and never sees the held-out fold."""
import numpy as np
import pandas as pd


def cutoff_for_retention(cal_scores: np.ndarray, retention: float) -> float:
    """Score above which a cardholder is flagged so that `retention` of the
    calibration partition is retained."""
    return float(np.quantile(cal_scores, retention))


def apply_cutoff(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(int)


def group_thresholds(cal_scores, cal_sex, cal_y, retention: float, n_grid: int = 400):
    """Search for per-sex thresholds that equalise the false-decline rate
    (FPR among non-defaulters) on the calibration partition while holding the
    overall retention at `retention`. Returns {sex_value: threshold}.

    The search is one-dimensional: for each candidate male threshold, the
    female threshold is set so that the total number flagged matches the
    target, and the pair with the smallest absolute FDR gap is kept."""
    cal_scores = np.asarray(cal_scores); cal_sex = np.asarray(cal_sex); cal_y = np.asarray(cal_y)
    n = len(cal_scores)
    n_flag_target = int(round((1 - retention) * n))
    m = cal_sex == 1
    s_m, s_f = np.sort(cal_scores[m])[::-1], np.sort(cal_scores[~m])[::-1]
    y_m, y_f = cal_y[m], cal_y[~m]
    nd_m, nd_f = (y_m == 0).sum(), (y_f == 0).sum()
    best = None
    for n_flag_m in np.linspace(0, min(n_flag_target, m.sum() - 1), n_grid).astype(int):
        n_flag_f = n_flag_target - n_flag_m
        if n_flag_f < 0 or n_flag_f >= (~m).sum():
            continue
        t_m = s_m[n_flag_m - 1] if n_flag_m > 0 else np.inf  # flag the top n_flag_m males
        t_f = s_f[n_flag_f - 1] if n_flag_f > 0 else np.inf
        # thresholds are inclusive at the boundary score: flagged = score >= t
        fdr_m = ((cal_scores[m] >= t_m) & (y_m == 0)).sum() / nd_m
        fdr_f = ((cal_scores[~m] >= t_f) & (y_f == 0)).sum() / nd_f
        gap = abs(fdr_m - fdr_f)
        if best is None or gap < best[0]:
            best = (gap, t_m, t_f, fdr_m, fdr_f)
    _, t_m, t_f, fdr_m, fdr_f = best
    return {1: float(t_m), 2: float(t_f)}, {"cal_fdr_male": fdr_m, "cal_fdr_female": fdr_f}


def apply_group_thresholds(scores, sex, thresholds: dict) -> np.ndarray:
    scores = np.asarray(scores); sex = np.asarray(sex)
    flag = np.zeros(len(scores), dtype=int)
    for s, t in thresholds.items():
        flag[sex == s] = (scores[sex == s] >= t).astype(int)
    return flag


def reweighing_weights(sex, y) -> np.ndarray:
    """Kamiran and Calders (2012): w(s, y) = P(s) P(y) / P(s, y), computed on
    the rows supplied (the fit partition)."""
    sex = np.asarray(sex); y = np.asarray(y)
    n = len(y)
    w = np.empty(n)
    for s in np.unique(sex):
        for yy in (0, 1):
            mask = (sex == s) & (y == yy)
            p_s, p_y, p_sy = (sex == s).mean(), (y == yy).mean(), mask.mean()
            w[mask] = p_s * p_y / p_sy
    return w


def reweighing_table(sex, y) -> pd.DataFrame:
    sex = np.asarray(sex); y = np.asarray(y)
    rows = []
    for s, lab in ((1, "male"), (2, "female")):
        for yy in (0, 1):
            mask = (sex == s) & (y == yy)
            rows.append({"sex": lab, "DEFAULT": yy, "n": int(mask.sum()),
                         "weight": round(((sex == s).mean() * (y == yy).mean()) / mask.mean(), 4)})
    return pd.DataFrame(rows)

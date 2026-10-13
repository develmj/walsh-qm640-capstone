"""Shared constants and helpers for the QM640 capstone pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 640
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "default of credit card clients.xls"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for p in (PROCESSED, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

TARGET = "DEFAULT"
PAY_COLS = [f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
MONEY_COLS = ["LIMIT_BAL"] + BILL_COLS + PAYAMT_COLS
DEMOG_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
ENGINEERED = ["UTILISATION", "PAY_RATIO", "MAX_DELAY"]

# Feature sets. The unaware set is the configuration audited in RQ3.
FEATURES_AWARE = MONEY_COLS + DEMOG_COLS + ["AGE"] + PAY_COLS + ENGINEERED
FEATURES_UNAWARE = [c for c in FEATURES_AWARE if c not in DEMOG_COLS]
NOMINAL = {"SEX": [1, 2], "EDUCATION": [0, 1, 2, 3, 4], "MARRIAGE": [0, 1, 2, 3]}

RETENTION_PRIMARY = 0.75
RETENTION_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]
N_BOOT = 1000
N_FOLDS = 5


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "clients_clean.parquet")


def load_folds() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "folds.csv")


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------------------
# DeLong (1988) AUC variance and paired test, fast implementation after
# Sun & Xu (2014). Positive class is y == 1.
# ---------------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_components(scores, y):
    """Return AUC and the structural components (V10, V01) for a score vector."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    pos = scores[y == 1]
    neg = scores[y == 0]
    m, n = len(pos), len(neg)
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return auc, v10, v01


def delong_ci(scores, y, alpha=0.05):
    from scipy.stats import norm
    auc, v10, v01 = delong_components(scores, y)
    var = v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01)
    se = np.sqrt(var)
    z = norm.ppf(1 - alpha / 2)
    return auc, se, auc - z * se, auc + z * se


def delong_paired(scores_a, scores_b, y):
    """Paired DeLong test for AUC_a - AUC_b on the same observations.

    Returns (auc_a, auc_b, diff, se_diff, z, p_two_sided, corr)."""
    from scipy.stats import norm
    auc_a, v10a, v01a = delong_components(scores_a, y)
    auc_b, v10b, v01b = delong_components(scores_b, y)
    m, n = len(v10a), len(v01a)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    cov = s10 / m + s01 / n
    var_a, var_b, cov_ab = cov[0, 0], cov[1, 1], cov[0, 1]
    var_diff = var_a + var_b - 2 * cov_ab
    se = np.sqrt(var_diff)
    diff = auc_a - auc_b
    z = diff / se
    p = 2 * norm.sf(abs(z))
    corr = cov_ab / np.sqrt(var_a * var_b)
    return auc_a, auc_b, diff, se, z, p, corr


def ks_statistic(scores, y):
    from scipy.stats import ks_2samp
    scores = np.asarray(scores)
    y = np.asarray(y).astype(int)
    return ks_2samp(scores[y == 1], scores[y == 0]).statistic


def brier(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))

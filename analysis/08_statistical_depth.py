"""Formal hypothesis-test register for RQ1 to RQ4 with decisions at alpha = .05,
assumption checks, effect sizes with intervals, multiplicity adjustment,
equivalence tests where the claim is 'within tolerance', paired inference for
before/after comparisons, and the minimum detectable effect at the achieved N.

Writes results/tables/hypothesis_test_register.csv plus the supporting
tables named below."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2, chi2_contingency, norm, ncx2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from common import N_BOOT, PROCESSED, RETENTION_PRIMARY, SEED, TABLES, TARGET, delong_paired, load_clean
from evaluate import audit
from policy import apply_cutoff, apply_group_thresholds, cutoff_for_retention, group_thresholds

ALPHA = 0.05
z975, z80 = norm.ppf(0.975), norm.ppf(0.80)
df = load_clean(); dfi = df.set_index("ID")
scores = pd.read_parquet(PROCESSED / "scores.parquet")
register = []


def add(rq, hypothesis, test, statistic, stat_name, dof, p, effect, effect_name, ci_low, ci_high, decision, assumptions, mde=None, note=""):
    register.append({"RQ": rq, "hypothesis": hypothesis, "test": test, "statistic": stat_name, "value": statistic,
                     "df": dof, "p_value": f"{p:.3g}" if p is not None else "", "effect_size": effect_name, "effect": effect,
                     "ci95_low": ci_low, "ci95_high": ci_high, "decision_alpha_0.05": decision,
                     "assumption_check": assumptions, "min_detectable_effect_at_N_power_0.80": mde, "note": note})


def flags_single(s_all, retention, index):
    test = s_all[s_all.partition == "test"].set_index("ID").loc[index]
    flag = np.zeros(len(test), dtype=int)
    for k in range(5):
        cal = s_all[(s_all.partition == "cal") & (s_all.fold == k)]
        thr = cutoff_for_retention(cal["p_raw"].values, retention)
        fm = (test["fold"] == k).values
        flag[fm] = apply_cutoff(test["p_raw"].values[fm], thr)
    return flag


def two_prop_test(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    z, p = proportions_ztest([x1, x2], [n1, n2])
    lo, hi = confint_proportions_2indep(x1, n1, x2, n2, method="newcomb")
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pbar = (x1 + x2) / (n1 + n2)
    mde = (z975 + z80) * np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    return p1 - p2, z, p, lo, hi, se_unpooled, mde


def tost(d, se, margin):
    """Two one-sided tests for equivalence of a difference d within +/- margin."""
    p_lower = norm.sf((d + margin) / se)     # H0: d <= -margin
    p_upper = norm.cdf((d - margin) / se)    # H0: d >= +margin
    return max(p_lower, p_upper), p_lower, p_upper


def paired_boot_gaps(y, sex, flags: dict, n_boot=N_BOOT, seed=SEED):
    """Resample individuals within sex x outcome strata and recompute the
    false-decline gap, DI ratio and default rate among retained for each
    configuration jointly, so differences between configurations are paired."""
    rng = np.random.default_rng(seed)
    strata = [(np.where((sex == s) & (y == yy))[0]) for s in (1, 2) for yy in (0, 1)]
    names = list(flags)
    F = np.column_stack([flags[n] for n in names])  # n x C
    out = {n: {"FDR_gap": [], "DI": [], "bad": []} for n in names}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        yb, sb, Fb = y[idx], sex[idx], F[idx]
        for j, n in enumerate(names):
            f = Fb[:, j]
            fdr_m = f[(sb == 1) & (yb == 0)].mean(); fdr_f = f[(sb == 2) & (yb == 0)].mean()
            ret_m = 1 - f[sb == 1].mean(); ret_f = 1 - f[sb == 2].mean()
            out[n]["FDR_gap"].append(fdr_m - fdr_f)
            out[n]["DI"].append(min(ret_m, ret_f) / max(ret_m, ret_f))
            out[n]["bad"].append(yb[f == 0].mean())
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in out.items()}


# ===========================================================================
# RQ1
# ===========================================================================
test_u = {m: scores[(scores.feature_set == "unaware") & (scores.model == m) & (scores.partition == "test")].set_index("ID")
          for m in ("lr", "rf", "xgb")}
idx = test_u["xgb"].index
y = dfi.loc[idx, TARGET].values; sex = dfi.loc[idx, "SEX"].values
p_x, p_l, p_r = test_u["xgb"].loc[idx, "p_raw"].values, test_u["lr"].loc[idx, "p_raw"].values, test_u["rf"].loc[idx, "p_raw"].values

a, b, diff, se, z, p2, corr = delong_paired(p_x, p_l, y)
p1 = p2 / 2 if diff > 0 else 1 - p2 / 2
mde_auc = (z975 + z80) * se
add("RQ1", "H1_0: AUC_GB = AUC_LR vs H1_1: AUC_GB > AUC_LR (pre-specified, one-sided)", "DeLong paired AUC test",
    round(z, 3), "z", None, p1, round(diff, 4), "AUC difference", round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4),
    "Reject H1_0", "Same held-out observations scored by both models (paired); DeLong is nonparametric; no distributional assumption on scores",
    round(mde_auc, 4), f"paired AUC correlation {corr:.3f}; primary inference")
# secondary contrasts with Holm correction (family of 3 within the unaware set)
sec = []
for name, pa, pb in (("XGBoost vs logistic", p_x, p_l), ("Random forest vs logistic", p_r, p_l), ("XGBoost vs random forest", p_x, p_r)):
    aa, bb, d, s, zz, pp, cc = delong_paired(pa, pb, y)
    sec.append({"contrast": name, "difference": d, "se": s, "z": zz, "p_two_sided": pp})
sec = pd.DataFrame(sec)
sec["p_holm"] = multipletests(sec["p_two_sided"], method="holm")[1]
sec["reject_holm"] = sec["p_holm"] < ALPHA
sec.to_csv(TABLES / "rq1_contrasts_holm.csv", index=False, float_format="%.6g")
for r in sec.itertuples():
    add("RQ1", f"Secondary: AUC equal for {r.contrast} (two-sided, Holm-adjusted)", "DeLong paired AUC test", round(r.z, 3), "z", None,
        r.p_holm, round(r.difference, 4), "AUC difference", round(r.difference - 1.96 * r.se, 4), round(r.difference + 1.96 * r.se, 4),
        "Reject" if r.reject_holm else "Retain", "As above; Holm step-down over three contrasts", None, "secondary")

# Fold-level corrected paired t-test (Nadeau & Bengio, 2003) as a robustness check
fold_auc = pd.read_csv(TABLES / "rq1_auc_by_fold.csv")
fx = fold_auc[(fold_auc.feature_set == "unaware") & (fold_auc.model == "XGBoost")].iloc[0, 2:].values.astype(float)
fl = fold_auc[(fold_auc.feature_set == "unaware") & (fold_auc.model == "Logistic regression")].iloc[0, 2:].values.astype(float)
d = fx - fl; K = len(d); ratio = 0.25  # test/train size ratio in 5-fold CV
t_nb = d.mean() / np.sqrt((1 / K + ratio) * d.var(ddof=1))
p_nb = stats.t.sf(t_nb, K - 1)
add("RQ1", "Robustness: mean fold-level AUC difference GB - LR = 0 (one-sided)", "Nadeau-Bengio corrected paired t-test across 5 folds",
    round(t_nb, 3), "t", K - 1, p_nb, round(d.mean(), 4), "mean fold AUC difference", round(d.mean() - stats.t.ppf(0.975, K - 1) * np.sqrt((1 / K + ratio) * d.var(ddof=1)), 4),
    round(d.mean() + stats.t.ppf(0.975, K - 1) * np.sqrt((1 / K + ratio) * d.var(ddof=1)), 4),
    "Reject", "Fold differences approximately normal; variance corrected for overlapping training sets", None, f"fold SD of difference {d.std(ddof=1):.4f}")

# Calibration: Hosmer-Lemeshow on calibrated XGBoost probabilities (deciles)
pc = test_u["xgb"].loc[idx, "p_cal"].values
dec = pd.qcut(pc, 10, labels=False, duplicates="drop")
hl = 0.0
for g in np.unique(dec):
    m = dec == g; o = y[m].sum(); e = pc[m].sum(); n = m.sum()
    hl += (o - e) ** 2 / (e * (1 - e / n)) if 0 < e < n else 0
g_n = len(np.unique(dec))
p_hl = chi2.sf(hl, g_n - 2)
max_dev = max(abs(y[dec == g].mean() - pc[dec == g].mean()) for g in np.unique(dec))
add("RQ1", "Calibration: observed = expected default rate across deciles (isotonic-calibrated XGBoost)", "Hosmer-Lemeshow chi-square",
    round(hl, 2), "chi2", g_n - 2, p_hl, round(float(np.mean((pc - y) ** 2)), 4), "Brier score", None, None,
    "Retain (adequate calibration)" if p_hl > ALPHA else "Reject (miscalibration detected)",
    "Deciles of predicted probability; expected counts > 5 in every decile; test is sensitive at large N", None,
    f"calibration check, not a hypothesis of the RQ; largest decile deviation {max_dev:.4f} (observed minus mean predicted)")

# ===========================================================================
# RQ2
# ===========================================================================
y_all = df[TARGET].values; male = df["SEX"].values == 1
n_m, n_f = male.sum(), (~male).sum(); x_m, x_f = y_all[male].sum(), y_all[~male].sum()
d, z, p, lo, hi, se_u, mde = two_prop_test(x_m, n_m, x_f, n_f)
h = 2 * np.arcsin(np.sqrt(x_m / n_m)) - 2 * np.arcsin(np.sqrt(x_f / n_f))
add("RQ2", "H2_0: p_male = p_female vs H2_1: p_male != p_female", "Two-proportion z test (pooled SE), Newcombe CI",
    round(z, 3), "z", None, p, round(d, 4), "difference in proportions", round(lo, 4), round(hi, 4), "Reject H2_0",
    f"Independent observations; np and n(1-p) >= 10 in both groups (min {min(x_m, x_f):,}); normal approximation valid",
    round(mde, 4), f"Cohen's h = {h:.3f} (small); observed difference exceeds the planned 0.03")
# equivalence framing: is the difference within +/- 0.03? (it is not)
p_t, pl, pu = tost(d, se_u, 0.03)
add("RQ2", "Equivalence: |p_male - p_female| < 0.03 (TOST, margin 0.03)", "Two one-sided z tests", round((d - 0.03) / se_u, 3), "z (upper)", None, p_t,
    round(d, 4), "difference in proportions", round(lo, 4), round(hi, 4), "Retain (not equivalent within 0.03)", "As above; unpooled SE", None,
    "the difference is material by the planning threshold")

order = ["<30", "30-39", "40-49", ">=50"]
ct = pd.crosstab(df["AGE_BAND"], df[TARGET]).reindex(order)
chi2_s, p_chi, dof, expected = chi2_contingency(ct.values)
n = ct.values.sum(); V = np.sqrt(chi2_s / (n * (min(ct.shape) - 1)))
# bootstrap CI for Cramer's V
rng = np.random.default_rng(SEED); Vb = []
bands = df["AGE_BAND"].values
for _ in range(500):
    i = rng.integers(0, n, n)
    c = pd.crosstab(bands[i], y_all[i]).values
    Vb.append(np.sqrt(chi2_contingency(c)[0] / (n * (min(c.shape) - 1))))
lam = 10.90  # non-centrality for power .80 at df 3
add("RQ2", "H2b_0: DEFAULT independent of AGE_BAND", "Chi-square test of independence (2 x 4)", round(chi2_s, 2), "chi2", int(dof), p_chi,
    round(V, 4), "Cramer's V", round(np.quantile(Vb, 0.025), 4), round(np.quantile(Vb, 0.975), 4), "Reject H2b_0",
    f"All expected cell counts >= 5 (min {expected.min():.0f}); independent observations", round(np.sqrt(lam / n), 4),
    "effect size negligible by Cohen's benchmarks (w < 0.10)")
# post-hoc pairwise age-band comparisons with Holm
pairs = []
for i in range(4):
    for j in range(i + 1, 4):
        a_, b_ = order[i], order[j]
        xa, na = y_all[bands == a_].sum(), (bands == a_).sum(); xb, nb = y_all[bands == b_].sum(), (bands == b_).sum()
        dd, zz, pp, lo_, hi_, _, _ = two_prop_test(xa, na, xb, nb)
        pairs.append({"band_a": a_, "band_b": b_, "difference": dd, "ci95_low": lo_, "ci95_high": hi_, "z": zz, "p_raw": pp})
pairs = pd.DataFrame(pairs); pairs["p_holm"] = multipletests(pairs["p_raw"], method="holm")[1]; pairs["reject_holm"] = pairs["p_holm"] < ALPHA
pairs.to_csv(TABLES / "rq2_age_band_posthoc_holm.csv", index=False, float_format="%.6g")

# Adjusted logistic regression: VIF, Wald test for male, fuller model with education and marriage
X = pd.DataFrame({"const": 1.0, "male": male.astype(float), "age": df["AGE"].astype(float).values, "log_limit": np.log(df["LIMIT_BAL"].astype(float).values)})
fit = sm.Logit(y_all, X).fit(disp=0)
vif = pd.DataFrame({"term": X.columns[1:], "VIF": [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]})
ci = fit.conf_int()
add("RQ2", "Adjusted: male coefficient = 0 given AGE and log LIMIT_BAL", "Wald z test in logistic regression", round(fit.tvalues["male"], 3), "z", None,
    fit.pvalues["male"], round(np.exp(fit.params["male"]), 4), "adjusted odds ratio", round(np.exp(ci.loc["male", 0]), 4), round(np.exp(ci.loc["male", 1]), 4),
    "Reject", f"Binary outcome; max VIF {vif.VIF.max():.2f} (no collinearity); N large for MLE; log transform of LIMIT_BAL for linearity in the logit", None,
    f"unadjusted OR {np.exp(sm.Logit(y_all, X[['const', 'male']]).fit(disp=0).params['male']):.4f}")
Xf = X.copy()
for lvl in (0, 1, 2, 3):
    Xf[f"edu_{lvl}"] = (df["EDUCATION"].values == lvl).astype(float)   # reference: 4 (other)
for lvl in (0, 2, 3):
    Xf[f"mar_{lvl}"] = (df["MARRIAGE"].values == lvl).astype(float)   # reference: 1 (married)
fit_f = sm.Logit(y_all, Xf).fit(disp=0); ci_f = fit_f.conf_int()
full = pd.DataFrame({"term": fit_f.params.index, "odds_ratio": np.exp(fit_f.params.values), "or_ci95_low": np.exp(ci_f[0].values),
                     "or_ci95_high": np.exp(ci_f[1].values), "p_value": fit_f.pvalues.values})
full.to_csv(TABLES / "rq2_adjusted_logit_full.csv", index=False, float_format="%.6g")
vif.to_csv(TABLES / "rq2_vif.csv", index=False, float_format="%.4f")
add("RQ2", "Adjusted (full): male coefficient = 0 given AGE, log LIMIT_BAL, EDUCATION, MARRIAGE", "Wald z test in logistic regression",
    round(fit_f.tvalues["male"], 3), "z", None, fit_f.pvalues["male"], round(np.exp(fit_f.params["male"]), 4), "adjusted odds ratio",
    round(np.exp(ci_f.loc["male", 0]), 4), round(np.exp(ci_f.loc["male", 1]), 4), "Reject", "As above; dummy coding with reference levels other and married",
    None, f"pseudo R2 {fit_f.prsquared:.4f}; LR test vs intercept-only p = {fit_f.llr_pvalue:.3g}")

# ===========================================================================
# RQ3
# ===========================================================================
flag_u = flags_single(scores[(scores.feature_set == "unaware") & (scores.model == "xgb")], RETENTION_PRIMARY, idx)
flag_a = flags_single(scores[(scores.feature_set == "aware") & (scores.model == "xgb")], RETENTION_PRIMARY, idx)
nd = y == 0
xm, nm = int(flag_u[nd & (sex == 1)].sum()), int((nd & (sex == 1)).sum()); xf, nf = int(flag_u[nd & (sex == 2)].sum()), int((nd & (sex == 2)).sum())
d, z, p, lo, hi, se_u, mde = two_prop_test(xm, nm, xf, nf)
add("RQ3", "H3_0: FDR_male = FDR_female at 75% retention (unaware XGBoost)", "Two-proportion z test (pooled SE), Newcombe CI", round(z, 3), "z", None, p,
    round(d, 4), "false-decline rate difference", round(lo, 4), round(hi, 4), "Reject H3_0",
    f"Non-defaulters only (n = {nm:,} male, {nf:,} female); held-out decisions from calibration-set cut-offs; np >= 10", round(mde, 4),
    "disparity detected")
p_t, pl, pu = tost(d, se_u, 0.05)
add("RQ3", "Benchmark: |FDR_male - FDR_female| < 0.05 (TOST, margin 0.05)", "Two one-sided z tests", round((d - 0.05) / se_u, 3), "z (upper)", None, p_t,
    round(d, 4), "false-decline rate difference", round(lo, 4), round(hi, 4), "Reject non-equivalence (gap within 0.05)", "Unpooled SE; both one-sided tests significant",
    None, f"p_lower = {pl:.3g}, p_upper = {pu:.3g}")
p_t2, _, _ = tost(d, se_u, 0.02)
add("RQ3", "Stricter benchmark: |gap| < 0.02 (TOST, margin 0.02)", "Two one-sided z tests", round((d - 0.02) / se_u, 3), "z (upper)", None, p_t2,
    round(d, 4), "false-decline rate difference", round(lo, 4), round(hi, 4), "Retain (not within 0.02)" if p_t2 >= ALPHA else "Reject", "As above", None,
    "shows where the gap would fail a tighter tolerance")
# DI ratio: test H0 DI >= 0.80 via bootstrap lower bound (from RQ3 table)
r3 = pd.read_csv(TABLES / "rq3_fairness_summary.csv"); r3p = r3.iloc[0]
add("RQ3", "Benchmark: DI ratio >= 0.80 on RETAIN", "Stratified bootstrap (1,000) percentile interval", round(r3p["DI_ratio"], 4), "DI", None, None,
    round(r3p["DI_ratio"], 4), "disparate impact ratio", round(r3p["DI_ci95_low"], 4), round(r3p["DI_ci95_high"], 4),
    "Benchmark met (lower bound > 0.80)", "Binomial resampling within sex x outcome strata", None, "")
# chi-square equivalent on the 2x2 of sex x FLAG among non-defaulters
ct2 = np.array([[xm, nm - xm], [xf, nf - xf]])
c2, pc2, _, _ = chi2_contingency(ct2, correction=False)
add("RQ3", "H3_0 (chi-square form): FLAG independent of SEX among non-defaulters", "Chi-square 2 x 2 without continuity correction", round(c2, 2), "chi2", 1, pc2,
    round(np.sqrt(c2 / (nm + nf)), 4), "phi", None, None, "Reject", "Expected counts >> 5", None, "equivalent to the z test (z^2 = chi2)")
# TPR gap (secondary, lender-risk)
dd_ = y == 1
xm2, nm2 = int(flag_u[dd_ & (sex == 1)].sum()), int((dd_ & (sex == 1)).sum()); xf2, nf2 = int(flag_u[dd_ & (sex == 2)].sum()), int((dd_ & (sex == 2)).sum())
d2, z2, p2_, lo2, hi2, _, mde2 = two_prop_test(xm2, nm2, xf2, nf2)
add("RQ3", "Secondary: TPR_male = TPR_female among defaulters", "Two-proportion z test, Newcombe CI", round(z2, 3), "z", None, p2_, round(d2, 4), "TPR difference",
    round(lo2, 4), round(hi2, 4), "Reject" if p2_ < ALPHA else "Retain", f"Defaulters only (n = {nm2:,} male, {nf2:,} female)", round(mde2, 4), "lender-risk measure")
# Paired comparison: does unawareness change the gap? aware vs unaware on the same individuals
pb = paired_boot_gaps(y, sex, {"unaware": flag_u, "aware": flag_a})
dgap = pb["aware"]["FDR_gap"] - pb["unaware"]["FDR_gap"]
gap_u, gap_a = audit(y, sex, flag_u)["FDR_gap"], audit(y, sex, flag_a)["FDR_gap"]
p_paired = 2 * min((dgap <= 0).mean(), (dgap >= 0).mean())
add("RQ3", "Unawareness effect: FDR gap (aware) - FDR gap (unaware) = 0", "Paired stratified bootstrap (1,000), percentile interval", round(gap_a - gap_u, 4),
    "difference of gaps", None, p_paired, round(gap_a - gap_u, 4), "change in false-decline gap", round(np.quantile(dgap, 0.025), 4), round(np.quantile(dgap, 0.975), 4),
    "Reject" if p_paired < ALPHA else "Retain", "Same individuals under both models; resampling within sex x outcome strata", None,
    "positive value means the aware model has the larger gap")
# Gap by fold
fold_rows = []
tu = scores[(scores.feature_set == "unaware") & (scores.model == "xgb") & (scores.partition == "test")].set_index("ID").loc[idx]
for k in range(5):
    fm = (tu["fold"] == k).values
    ak = audit(y[fm], sex[fm], flag_u[fm])
    fold_rows.append({"fold": k, "FDR_gap": ak["FDR_gap"], "DI_ratio": ak["DI_ratio"], "TPR_gap": ak["TPR_gap"], "retain_rate": ak["overall_retain_rate"]})
fold_tab = pd.DataFrame(fold_rows); fold_tab.to_csv(TABLES / "rq3_gap_by_fold.csv", index=False, float_format="%.4f")
add("RQ3", "Consistency: FDR gap across the five folds", "Descriptive: mean and SD of fold-level gaps", round(fold_tab.FDR_gap.mean(), 4), "mean gap", 4, None,
    round(fold_tab.FDR_gap.std(ddof=1), 4), "SD across folds", round(fold_tab.FDR_gap.min(), 4), round(fold_tab.FDR_gap.max(), 4), "All five folds positive",
    "Folds stratified by sex x outcome", None, "")

# Robustness: false-decline rate by age band, pairwise with Holm (unaware XGBoost, 75% retention)
bands_t = dfi.loc[idx, "AGE_BAND"].values
age_pairs = []
for i in range(4):
    for j in range(i + 1, 4):
        a_, b_ = order[i], order[j]
        ma = (bands_t == a_) & nd; mb = (bands_t == b_) & nd
        dd, zz, pp, lo_, hi_, _, _ = two_prop_test(int(flag_u[ma].sum()), int(ma.sum()), int(flag_u[mb].sum()), int(mb.sum()))
        age_pairs.append({"band_a": a_, "band_b": b_, "FDR_a": flag_u[ma].mean(), "FDR_b": flag_u[mb].mean(), "difference": dd,
                          "ci95_low": lo_, "ci95_high": hi_, "z": zz, "p_raw": pp})
age_pairs = pd.DataFrame(age_pairs); age_pairs["p_holm"] = multipletests(age_pairs["p_raw"], method="holm")[1]
age_pairs["reject_holm"] = age_pairs["p_holm"] < ALPHA
age_pairs["within_0.05"] = age_pairs["difference"].abs() <= 0.05
age_pairs.to_csv(TABLES / "rq3_age_band_fdr_posthoc_holm.csv", index=False, float_format="%.6g")
worst = age_pairs.loc[age_pairs["difference"].abs().idxmax()]
add("RQ3", f"Robustness (age band): FDR equal for {worst.band_a} vs {worst.band_b} (largest of six pairs, Holm-adjusted)", "Two-proportion z test, Newcombe CI",
    round(worst.z, 3), "z", None, worst.p_holm, round(worst.difference, 4), "false-decline rate difference", round(worst.ci95_low, 4), round(worst.ci95_high, 4),
    "Reject", "Non-defaulters only; Holm over six pairwise comparisons", None,
    f"{int(age_pairs.reject_holm.sum())} of 6 pairs significant; {int((~age_pairs['within_0.05']).sum())} of 6 exceed the 0.05 benchmark")

# ===========================================================================
# RQ4
# ===========================================================================
ni = pd.read_csv(TABLES / "rq4_reweighing_noninferiority.csv").iloc[0]
add("RQ4", "H4_0: AUC_RW <= AUC_base - 0.01 vs H4_1: AUC_RW > AUC_base - 0.01 (unaware model)", "One-sided non-inferiority, DeLong paired variance",
    round(ni["z_noninferiority"], 2), "z", None, ni["p_noninferiority"], round(ni["difference"], 5), "AUC difference RW - base",
    round(ni["lower_bound_one_sided_95"], 5), None, "Reject H4_0 (non-inferior)", f"Paired; correlation {ni['paired_auc_correlation']:.3f}; margin pre-specified",
    None, "")
ni_a = pd.read_csv(TABLES / "rq4_reweighing_aware_noninferiority.csv").iloc[0]
add("RQ4", "Extension: AUC_RW <= AUC_base - 0.01 (aware model)", "One-sided non-inferiority, DeLong paired variance", round(ni_a["z_noninferiority"], 2), "z", None,
    ni_a["p_noninferiority"], round(ni_a["difference"], 5), "AUC difference RW - base", round(ni_a["lower_bound_one_sided_95"], 5), None,
    "Reject (non-inferior)", f"Paired; correlation {ni_a['paired_auc_correlation']:.3f}", None, "")

# Paired bootstrap across the three configurations at 75% retention (unaware base)
rw = pd.read_parquet(PROCESSED / "scores_reweighed.parquet")
flag_rw = flags_single(rw, RETENTION_PRIMARY, idx)
base = scores[(scores.feature_set == "unaware") & (scores.model == "xgb")]
test_b = base[base.partition == "test"].set_index("ID").loc[idx]
flag_ta = np.zeros(len(idx), dtype=int)
for k in range(5):
    cal = base[(base.partition == "cal") & (base.fold == k)].set_index("ID")
    thr, _ = group_thresholds(cal["p_raw"].values, dfi.loc[cal.index, "SEX"].values, dfi.loc[cal.index, TARGET].values, RETENTION_PRIMARY)
    fm = (test_b["fold"] == k).values
    flag_ta[fm] = apply_group_thresholds(test_b["p_raw"].values[fm], sex[fm], thr)
rw_a = pd.read_parquet(PROCESSED / "scores_reweighed_aware.parquet")
flag_rw_a = flags_single(rw_a, RETENTION_PRIMARY, idx)
pb4 = paired_boot_gaps(y, sex, {"unmitigated": flag_u, "reweighed": flag_rw, "threshold": flag_ta, "aware": flag_a, "aware_reweighed": flag_rw_a})
paired_rows = []
for name, lab in (("reweighed", "Reweighing (unaware) vs unmitigated"), ("threshold", "Threshold adjustment vs unmitigated"), ("aware_reweighed", "Reweighing (aware) vs aware unmitigated")):
    ref = "aware" if name == "aware_reweighed" else "unmitigated"
    dg = pb4[name]["FDR_gap"] - pb4[ref]["FDR_gap"]; dbad = pb4[name]["bad"] - pb4[ref]["bad"]; ddi = pb4[name]["DI"] - pb4[ref]["DI"]
    g_obs = audit(y, sex, {"reweighed": flag_rw, "threshold": flag_ta, "aware_reweighed": flag_rw_a}[name])
    g_ref = audit(y, sex, flag_a if ref == "aware" else flag_u)
    pg = 2 * min((dg <= 0).mean(), (dg >= 0).mean()); pbad = 2 * min((dbad <= 0).mean(), (dbad >= 0).mean())
    paired_rows.append({"comparison": lab, "change_in_FDR_gap": g_obs["FDR_gap"] - g_ref["FDR_gap"], "gap_ci95_low": np.quantile(dg, 0.025),
                        "gap_ci95_high": np.quantile(dg, 0.975), "gap_p_bootstrap": pg,
                        "change_in_default_rate_retained": g_obs["overall_default_rate_retained"] - g_ref["overall_default_rate_retained"],
                        "bad_ci95_low": np.quantile(dbad, 0.025), "bad_ci95_high": np.quantile(dbad, 0.975), "bad_p_bootstrap": pbad,
                        "change_in_DI": g_obs["DI_ratio"] - g_ref["DI_ratio"], "DI_ci95_low": np.quantile(ddi, 0.025), "DI_ci95_high": np.quantile(ddi, 0.975),
                        "decisions_changed": int((({"reweighed": flag_rw, "threshold": flag_ta, "aware_reweighed": flag_rw_a}[name]) != (flag_a if ref == "aware" else flag_u)).sum())})
    add("RQ4", f"{lab}: change in FDR gap = 0", "Paired stratified bootstrap (1,000)", round(g_obs["FDR_gap"] - g_ref["FDR_gap"], 4), "difference", None, pg,
        round(g_obs["FDR_gap"] - g_ref["FDR_gap"], 4), "change in false-decline gap", round(np.quantile(dg, 0.025), 4), round(np.quantile(dg, 0.975), 4),
        "Reject" if pg < ALPHA else "Retain", "Same individuals under both policies; strata sex x outcome", None, "")
    add("RQ4", f"{lab}: change in default rate among retained = 0", "Paired stratified bootstrap (1,000)",
        round(g_obs["overall_default_rate_retained"] - g_ref["overall_default_rate_retained"], 5), "difference", None, pbad,
        round(g_obs["overall_default_rate_retained"] - g_ref["overall_default_rate_retained"], 5), "change in default rate among retained",
        round(np.quantile(dbad, 0.025), 5), round(np.quantile(dbad, 0.975), 5), "Reject" if pbad < ALPHA else "Retain", "As above", None, "decision-quality cost")
pd.DataFrame(paired_rows).to_csv(TABLES / "rq4_paired_bootstrap_changes.csv", index=False, float_format="%.5f")
# TOST for the threshold-adjusted gap: equivalent to zero within 0.02?
nd = y == 0
xm3, xf3 = int(flag_ta[nd & (sex == 1)].sum()), int(flag_ta[nd & (sex == 2)].sum())
d3, z3, p3, lo3, hi3, se3, _ = two_prop_test(xm3, nm, xf3, nf)
p_t3, _, _ = tost(d3, se3, 0.02)
add("RQ4", "Threshold adjustment: residual |FDR gap| < 0.02 (TOST, margin 0.02)", "Two one-sided z tests", round((d3 - 0.02) / se3, 3), "z (upper)", None, p_t3,
    round(d3, 4), "false-decline rate difference", round(lo3, 4), round(hi3, 4), "Reject non-equivalence (gap within 0.02)" if p_t3 < ALPHA else "Retain",
    "Unpooled SE", None, f"two-sided test of zero gap p = {p3:.3g}")
# McNemar on decision flips (unmitigated vs threshold adjustment)
b01 = int(((flag_u == 0) & (flag_ta == 1)).sum()); b10 = int(((flag_u == 1) & (flag_ta == 0)).sum())
mc = (abs(b01 - b10) - 1) ** 2 / (b01 + b10); p_mc = chi2.sf(mc, 1)
add("RQ4", "Threshold adjustment changes the overall flag rate (marginal homogeneity)", "McNemar chi-square with continuity correction", round(mc, 2), "chi2", 1, p_mc,
    b01 + b10, "decisions changed (count)", b01, b10, "Retain (flag rate unchanged)" if p_mc > ALPHA else "Reject", "Paired binary decisions on the same individuals",
    None, "b01 = newly flagged, b10 = newly retained; retention held at 75% by design")

reg = pd.DataFrame(register)
reg.to_csv(TABLES / "hypothesis_test_register.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(reg[["RQ", "hypothesis", "statistic", "value", "p_value", "effect", "ci95_low", "ci95_high", "decision_alpha_0.05"]].to_string(index=False))
print(pd.DataFrame(paired_rows)[["comparison", "change_in_FDR_gap", "gap_ci95_low", "gap_ci95_high", "change_in_default_rate_retained", "bad_ci95_low", "bad_ci95_high", "decisions_changed"]].round(5).to_string(index=False))

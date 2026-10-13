"""RQ2: do observed default rates differ by sex and age band, and how large are
the differences before and after adjustment for credit limit and age?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_effectsize, proportions_ztest

from common import TABLES, TARGET, load_clean

df = load_clean()
y = df[TARGET].values
male = df["SEX"].values == 1

# ---- Two-proportion z test, male vs female --------------------------------
n_m, n_f = male.sum(), (~male).sum()
d_m, d_f = y[male].sum(), y[~male].sum()
p_m, p_f = d_m / n_m, d_f / n_f
z, p_two = proportions_ztest([d_m, d_f], [n_m, n_f])
lo, hi = confint_proportions_2indep(d_m, n_m, d_f, n_f, method="wald")
h = proportion_effectsize(p_m, p_f)
rows = [{
    "comparison": "male vs female default rate",
    "n_male": int(n_m), "n_female": int(n_f),
    "rate_male": round(p_m, 4), "rate_female": round(p_f, 4),
    "difference": round(p_m - p_f, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
    "z": round(z, 3), "p_value": float(f"{p_two:.3g}"), "cohens_h": round(h, 3),
}]
pd.DataFrame(rows).to_csv(TABLES / "rq2_sex_two_proportion.csv", index=False)

# ---- Chi-square, default by age band ----------------------------------------
order = ["<30", "30-39", "40-49", ">=50"]
ct = pd.crosstab(df["AGE_BAND"], df[TARGET]).reindex(order)
chi2, p_chi, dof, expected = chi2_contingency(ct.values)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
age_rows = []
for band in order:
    sub = df[df["AGE_BAND"] == band][TARGET]
    k, m = int(sub.sum()), len(sub)
    se = np.sqrt(sub.mean() * (1 - sub.mean()) / m)
    age_rows.append({"age_band": band, "n": m, "defaults": k, "default_rate": round(sub.mean(), 4),
                     "ci95_low": round(sub.mean() - 1.96 * se, 4), "ci95_high": round(sub.mean() + 1.96 * se, 4)})
age_tab = pd.DataFrame(age_rows)
age_tab.to_csv(TABLES / "rq2_age_band_rates.csv", index=False)
pd.DataFrame([{"test": "chi-square independence, DEFAULT x AGE_BAND", "chi2": round(chi2, 2), "df": int(dof),
               "p_value": float(f"{p_chi:.3g}"), "cramers_v": round(cramers_v, 4), "min_expected": round(expected.min(), 1)}]
             ).to_csv(TABLES / "rq2_age_band_chisq.csv", index=False)

# ---- Adjusted logistic regression -------------------------------------------
X = pd.DataFrame({
    "const": 1.0,
    "male": male.astype(float),
    "age": df["AGE"].values.astype(float),
    "log_limit": np.log(df["LIMIT_BAL"].values.astype(float)),
})
fit = sm.Logit(y, X).fit(disp=0)
params, ci = fit.params, fit.conf_int()
adj = []
for term in ["male", "age", "log_limit"]:
    adj.append({"term": term, "coef": round(params[term], 4), "odds_ratio": round(np.exp(params[term]), 4),
                "or_ci95_low": round(np.exp(ci.loc[term, 0]), 4), "or_ci95_high": round(np.exp(ci.loc[term, 1]), 4),
                "p_value": float(f"{fit.pvalues[term]:.3g}")})
# unadjusted odds ratio for comparison
fit0 = sm.Logit(y, X[["const", "male"]]).fit(disp=0)
ci0 = fit0.conf_int()
adj.insert(0, {"term": "male (unadjusted)", "coef": round(fit0.params["male"], 4), "odds_ratio": round(np.exp(fit0.params["male"]), 4),
               "or_ci95_low": round(np.exp(ci0.loc["male", 0]), 4), "or_ci95_high": round(np.exp(ci0.loc["male", 1]), 4),
               "p_value": float(f"{fit0.pvalues['male']:.3g}")})
# adjusted marginal difference in predicted probability (average over sample)
X1, X0 = X.copy(), X.copy(); X1["male"] = 1.0; X0["male"] = 0.0
adj_diff = float(fit.predict(X1).mean() - fit.predict(X0).mean())
adj_tab = pd.DataFrame(adj)
adj_tab.to_csv(TABLES / "rq2_adjusted_logit.csv", index=False)
pd.DataFrame([{"quantity": "raw difference male - female", "value": round(p_m - p_f, 4)},
              {"quantity": "adjusted difference (AGE, log LIMIT_BAL), average marginal", "value": round(adj_diff, 4)}]
             ).to_csv(TABLES / "rq2_raw_vs_adjusted_difference.csv", index=False)

# Also education and marriage descriptive rates (context only)
for c in ["EDUCATION", "MARRIAGE"]:
    g = df.groupby(c)[TARGET].agg(["count", "mean"]).round(4)
    g.to_csv(TABLES / f"rq2_rate_by_{c.lower()}.csv")

print(pd.DataFrame(rows).T)
print(age_tab)
print(f"chi2={chi2:.1f} df={dof} p={p_chi:.3g} V={cramers_v:.4f}")
print(adj_tab)
print("adjusted marginal difference:", round(adj_diff, 4))

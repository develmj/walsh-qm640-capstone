# Accuracy and fairness trade-offs in machine learning credit default prediction

QM640 Data Analytics Capstone, Walsh College. Mrityunjay Ravi Iyer, Fall 2026.
Repository: https://github.com/develmj/walsh-qm640-capstone

A reproducible audit of three default-prediction models on the UCI Default of
Credit Card Clients dataset (Yeh, 2009), measuring how their simulated decisions
differ by sex at a 75% retention cut-off and what two mitigation methods cost.
Everything in `results/` is produced by code from the raw file with seed 640.

## Data

UCI Machine Learning Repository dataset 350, https://doi.org/10.24432/C55S3H.
30,000 Taiwanese credit-card holders observed April to September 2005; the
target is a missed payment in October 2005 (22.12% prevalence). Licence CC BY 4.0.
`src/download.py` fetches the file (with a mirror fallback), verifies the
SHA-256 `c76095b2…9e44`, and checks the documented structure. The data
dictionary is `data/data_dictionary.csv`. The raw file is not committed.

## Run

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all            # about 7 minutes on 2 cores; XGB_ITER=50 widens the XGBoost search
```

`make all` runs, in order: `data` (download, recode, folds), `rq2` (EDA and
RQ2 tests), `train` (fit, calibrate, score), `rq1`, `rq3`, `rq4`,
`sensitivity` (coding sensitivities and SHAP), `extensions` (ThresholdOptimizer
cross-check, reweighing on the aware model, age-band audit), `register` (the
hypothesis-test register), `sample-size` (Table 1 of the synopsis). Each target can be run alone. Notebooks in `notebooks/` wrap the
same scripts for interactive use.

## Layout

```
src/        download.py  prepare.py  train.py  policy.py  evaluate.py  sample_size.py  common.py
analysis/   01_eda  02_rq2  03_rq1  04_rq3  05_rq4  06_sensitivity_shap  07_rq4_extensions  08_statistical_depth
notebooks/  the same eight steps as notebooks
data/       raw/ (fetched)  processed/ (parquet, folds.csv, scores.parquet)  data_dictionary.csv
results/    tables/ (CSV, one per analysis)  figures/ (PNG)
```

## Design

Stratified five-fold cross-validation keyed on ID. Inside each outer training
set, 80% is the fit partition (model training and inner-CV tuning) and 20% is
the calibration partition (isotonic recalibration, retention cut-offs,
per-sex thresholds, reweighing weights). The held-out fold is scored once.
Every cardholder therefore has one held-out score and one held-out decision
from a configuration that never saw them. The unaware feature set drops SEX,
EDUCATION and MARRIAGE; the aware set keeps them.

## Results (pooled held-out folds, N = 30,000)

Every test below is listed with its statistic, p value, effect size,
interval, assumption check and decision in
`results/tables/hypothesis_test_register.csv` (34 entries).

RQ1. XGBoost AUC 0.784 (95% CI 0.778 to 0.791) against logistic regression
0.755 (0.748 to 0.762). Pre-specified paired DeLong contrast: +0.029 (0.026 to
0.033), z = 15.7, one-sided p < .001; H1_0 rejected. A Nadeau-Bengio corrected
t-test across folds agrees (t(4) = 9.85, p < .001). Random forest ties XGBoost
(difference 0.0004, Holm-adjusted p = .62). Adding the demographic fields
raises XGBoost AUC by 0.0015. Brier after isotonic calibration 0.135 for both
ensembles, 0.143 for logistic; Hosmer-Lemeshow rejects at p = .015, with the
largest decile deviation under one percentage point, which is the test's
sensitivity at this N rather than a calibration fault. Minimum detectable AUC
difference at this N: 0.005.

RQ2. Men default at 24.17% and women at 20.78%: difference 3.39 points
(Newcombe 95% CI 2.42 to 4.36), z = 6.92, p < .001, Cohen's h = 0.08; H2_0
rejected, and a TOST shows the gap is not within the planned 0.03 margin.
Adjusting for age and log credit limit roughly halves the gap (adjusted OR
1.12, CI 1.06 to 1.18, z = 3.88; with education and marriage added, OR 1.14,
CI 1.07 to 1.20; max VIF 1.02). Age band is associated with default
(chi-square(3) = 44.1, p < .001) with a negligible effect (Cramér's V = 0.038,
bootstrap CI 0.027 to 0.050); five of six pairwise band comparisons survive
Holm correction. Median credit limit is NT$130,000 for men and NT$150,000 for
women.

RQ3. At 75% retention with the unaware XGBoost model, creditworthy men are
wrongly flagged at 17.6% against 14.8% for creditworthy women: a false-decline
gap of 2.79 points (Newcombe CI 1.82 to 3.77), z = 5.68, p < .001; H3_0
rejected, and the gap is positive in all five folds (SD 0.96 points). A TOST
with margin 0.05 rejects non-equivalence (p < .001), so the gap is inside the
benchmark; a TOST with margin 0.02 does not (p = .94). The disparate impact
ratio on retention is 0.946 (bootstrap CI 0.934 to 0.958), above 0.80. The
TPR gap is 2.4 points (p = .053). The aware model's gap is 1.03 points wider
(paired bootstrap CI 0.67 to 1.39, p < .001), so unawareness narrows the gap
without removing it. SHAP attributes the gap to repayment history (MAX_DELAY,
PAY_0) first and credit limit third. By age band, the over-50 group's
false-decline rate is 6.8 points above the 30 to 39 group (Holm-adjusted
p < .001), which exceeds the 0.05 benchmark; age disparity is larger than sex
disparity in this portfolio.

RQ4. Reweighing the unaware model is non-inferior (AUC 0.7839 vs 0.7841,
one-sided lower bound −0.0004 against a −0.01 margin, z = 54.1) but changes the
false-decline gap by −0.10 points (paired bootstrap CI −0.34 to +0.14,
p = .39; equivalent to zero within ±0.5 points by bootstrap TOST, p < .001):
the Kamiran-Calders weights lie between 0.92 and 1.06 and the model never sees
sex. Reweighing the aware model, where the weights can act
directly, cuts the gap by 1.84 points (CI −2.13 to −1.55, p < .001) at an AUC
cost of 0.0004 (non-inferior, z = 35.0). Per-sex threshold adjustment at 75%
retention cuts the gap by 2.36 points (CI −2.64 to −2.08, p < .001) to a
residual 0.43 points that a TOST places within 0.02 (p < .001), changing 357
of 30,000 decisions, with the default rate among retained cardholders up 0.03
points (CI −0.04 to +0.11, p = .38, within ±0.5 points by bootstrap TOST) and
the flag rate within ±0.5 points by paired TOST (change +0.04 points). Paired
bootstraps use 10,000 draws. Fairlearn's ThresholdOptimizer with a false-positive-rate parity
constraint reaches a 0.97-point gap at 70.2% retention; the direct search at
the same retention gives 0.82, so the two agree. The trade-off table across
65% to 85% retention is `results/tables/rq4_tradeoff_table.csv`.

Both coding sensitivities (collapsing PAY codes −2 and 0; merging undocumented
EDUCATION and MARRIAGE levels into other) change AUC and the gap by less than
0.001.

## Reproducibility across machines

The pipeline was run end to end on three environments: a cloud container
(Python 3.13, xgboost 3.4.1, scikit-learn 1.9.1, 2 cores), which is the
primary run behind the report, and an Ubuntu 22.04 Docker container on a Mac
Studio (Apple silicon, 4 cores) twice, once with the container's Python 3.10 and older
libraries (xgboost 3.2.0, scikit-learn 1.6.1, `requirements-py310.txt`) and
once in a Python 3.12 environment with the exact versions in
`requirements.txt` (created with `uv`). The comparison is in
`results/reproducibility_cross_machine.csv`.

With pinned versions the Mac run reproduces the primary run in full: every
results table and all 34 rows of the hypothesis-test register are identical
(same MD5 on the register file). With the older libraries, deterministic
quantities (logistic regression AUCs, every RQ2 statistic, fold assignment,
file checksum) still match to four decimals, XGBoost quantities differ by at
most 0.0005 in AUC and 0.003 in a fairness gap because the random search
selects different hyperparameters in some folds, and 32 of 34 register
decisions match. The two that differ are tests of a change equal to zero on
effects under a quarter of a point (reweighing's change in the unaware gap,
p = .39 vs .047; the bad-rate cost of threshold adjustment, p = .38 vs .028).
The equivalence tests at ±0.5 points, added for exactly this reason, give the
same decision on all three runs. Both Mac registers are stored as
`results/hypothesis_test_register_mac_pinned.csv` and
`results/hypothesis_test_register_mac_unpinned.csv`.

On a host with a short per-command time limit, `train.py` accepts a partial
run: `PART=unaware:0,1 python3 src/train.py` trains those folds, and
`PART=merge python3 src/train.py` assembles the full outputs. On Python 3.10
use `requirements-py310.txt` in place of the pinned file.

## Key output files

| File | Content |
|---|---|
| `results/tables/rq1_performance.csv` | AUC with DeLong CI, KS, Brier raw and calibrated, F1 at cut-off, per model and feature set |
| `results/tables/rq1_delong_contrasts.csv` | Paired DeLong contrasts with difference, CI, z, p, paired correlation |
| `results/tables/rq2_*.csv` | Two-proportion test, age-band chi-square, adjusted logistic regression |
| `results/tables/rq3_fairness_summary.csv` | DI ratio, false-decline and TPR gaps with bootstrap CIs, benchmark verdicts |
| `results/tables/rq3_confusion_by_sex.csv` | Per-sex confusion matrices for four configurations |
| `results/tables/rq4_tradeoff_table.csv` | Retention, default rate among retained, AUC, DI, gaps, at five retention rates for three configurations |
| `results/tables/rq4_reweighing_noninferiority.csv` | Paired-AUC non-inferiority test |
| `results/tables/rq_sensitivity_coding.csv` | Coding-assumption sensitivity runs |
| `results/tables/hypothesis_test_register.csv` | Every hypothesis test with statistic, df, p, effect size, CI, decision, assumption check, minimum detectable effect |
| `results/tables/rq4_paired_bootstrap_changes.csv` | Paired before/after changes in gap, DI, and bad rate with intervals |
| `results/tables/rq4_thresholdoptimizer_crosscheck.csv`, `rq4_reweighing_aware_model.csv` | RQ4 extensions |
| `results/tables/rq3_age_band_audit.csv`, `rq3_age_band_fdr_posthoc_holm.csv` | RQ3 age-band robustness |
| `results/figures/*.png` | EDA, ROC and reliability, fairness gaps, trade-off curves, SHAP by sex |

## Citation

Yeh, I.-C. (2009). Default of credit card clients [Data set]. UCI Machine
Learning Repository. https://doi.org/10.24432/C55S3H

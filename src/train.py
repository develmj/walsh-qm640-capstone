"""Train logistic regression, random forest, and XGBoost on the fit partition of
each outer fold, calibrate on the calibration partition, and score both the
calibration partition and the held-out fold. Run for the aware and unaware
feature sets. Writes data/processed/scores.parquet and
results/tables/rq1_xgb_best_params.csv.

Scores are stored for every (model, feature_set, fold, partition in {cal, test})
so that policy cut-offs can be set on 'cal' and evaluated on 'test'."""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import (FEATURES_AWARE, FEATURES_UNAWARE, MONEY_COLS, N_FOLDS, NOMINAL,
                    PROCESSED, SEED, TABLES, TARGET, load_clean, load_folds, signed_log1p)

warnings.filterwarnings("ignore")


def make_lr(features):
    money = [c for c in features if c in MONEY_COLS]
    nominal = [c for c in features if c in NOMINAL]
    other = [c for c in features if c not in money and c not in nominal]
    pre = ColumnTransformer([
        ("money", Pipeline([("log", FunctionTransformer(signed_log1p)), ("sc", StandardScaler())]), money),
        ("other", StandardScaler(), other),
        ("nom", OneHotEncoder(categories=[NOMINAL[c] for c in nominal], handle_unknown="ignore"), nominal),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"))])


def make_rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=2, random_state=SEED)


def make_xgb(spw):
    return XGBClassifier(objective="binary:logistic", tree_method="hist", n_jobs=2, random_state=SEED,
                         scale_pos_weight=spw, eval_metric="auc", verbosity=0)


XGB_SPACE = {
    "n_estimators": [200, 300, 400, 600],
    "learning_rate": [0.02, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10, 20],
    "reg_lambda": [1.0, 3.0, 10.0],
}


def run(feature_set: str, features: list, df: pd.DataFrame, folds: pd.DataFrame, sample_weight=None,
        xgb_params_override=None, models=("lr", "rf", "xgb"), tag=None):
    """Train on fit, calibrate on cal, score cal and test for each fold.

    sample_weight: optional array aligned with df rows (used for reweighing).
    xgb_params_override: dict fold -> params (skip tuning, used for reweighing).
    """
    tag = tag or feature_set
    X_all = df[features]
    y_all = df[TARGET].values
    out, best = [], []
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for k in range(N_FOLDS):
        part = folds[f"part_fold{k}"].values
        fit_idx = np.where(part == "fit")[0]
        cal_idx = np.where(part == "cal")[0]
        test_idx = np.where(part == "test")[0]
        if len(fit_idx) == 0:
            continue  # fold disabled by caller
        Xf, yf = X_all.iloc[fit_idx], y_all[fit_idx]
        wf = None if sample_weight is None else sample_weight[fit_idx]
        spw = (yf == 0).sum() / (yf == 1).sum()
        fitted = {}
        t0 = time.time()
        if "lr" in models:
            gs = GridSearchCV(make_lr(features), {"clf__C": [0.01, 0.1, 1.0, 10.0]}, cv=inner, scoring="roc_auc", n_jobs=1)
            gs.fit(Xf, yf, **({"clf__sample_weight": wf} if wf is not None else {}))
            fitted["lr"] = gs.best_estimator_
        if "rf" in models:
            rf = make_rf()
            rf.fit(Xf, yf, sample_weight=wf)
            fitted["rf"] = rf
        if "xgb" in models:
            if xgb_params_override is not None:
                params = dict(xgb_params_override[k])
                xgb = make_xgb(spw).set_params(**params)
                xgb.fit(Xf, yf, sample_weight=wf)
            else:
                rs = RandomizedSearchCV(make_xgb(spw), XGB_SPACE, n_iter=int(os.environ.get("XGB_ITER", 12)), cv=inner, scoring="roc_auc",
                                        random_state=SEED + k, n_jobs=1, refit=True)
                rs.fit(Xf, yf)
                xgb = rs.best_estimator_
                params = rs.best_params_
                best.append({"feature_set": tag, "fold": k, "inner_cv_auc": round(rs.best_score_, 4), **params})
            fitted["xgb"] = xgb
        for name, model in fitted.items():
            p_cal = model.predict_proba(X_all.iloc[cal_idx])[:, 1]
            p_test = model.predict_proba(X_all.iloc[test_idx])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_all[cal_idx])
            for part_name, idx, p in (("cal", cal_idx, p_cal), ("test", test_idx, p_test)):
                out.append(pd.DataFrame({
                    "ID": df["ID"].values[idx], "fold": k, "partition": part_name, "model": name,
                    "feature_set": tag, "p_raw": p, "p_cal": iso.predict(p),
                }))
        print(f"[{tag}] fold {k} done in {time.time() - t0:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True), pd.DataFrame(best)


def merge_parts():
    """Assemble scores.parquet and the XGBoost parameter files from per-part files."""
    parts = sorted((PROCESSED / "parts").glob("scores_*.parquet"))
    bests = sorted((PROCESSED / "parts").glob("best_*.json"))
    if len(parts) != 2 * N_FOLDS:
        raise SystemExit(f"expected {2 * N_FOLDS} score parts, found {len(parts)}")
    scores = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    best = pd.concat([pd.read_json(b) for b in bests], ignore_index=True).sort_values(["feature_set", "fold"], ascending=[False, True])
    scores.to_parquet(PROCESSED / "scores.parquet", index=False)
    best.to_csv(TABLES / "rq1_xgb_best_params.csv", index=False)
    best.to_json(PROCESSED / "xgb_best_params.json", orient="records")
    print(scores.groupby(["feature_set", "model", "partition"]).size())


if __name__ == "__main__":
    # Optional partial run for hosts with short command limits:
    #   PART=unaware:0,1 python3 src/train.py   trains those folds only and writes
    #   data/processed/parts/; PART=merge assembles the full outputs.
    part = os.environ.get("PART")
    if part == "merge":
        merge_parts()
        raise SystemExit(0)
    df = load_clean()
    folds = load_folds().set_index("ID").loc[df["ID"]].reset_index()
    assert (folds["ID"].values == df["ID"].values).all()
    if part:
        fs, fold_list = part.split(":")
        features = FEATURES_UNAWARE if fs == "unaware" else FEATURES_AWARE
        (PROCESSED / "parts").mkdir(exist_ok=True)
        for k in [int(x) for x in fold_list.split(",")]:
            folds_k = folds.copy()
            for j in range(N_FOLDS):
                if j != k:
                    folds_k[f"part_fold{j}"] = "skip"
            s, b = run(fs, features, df, folds_k)
            s.to_parquet(PROCESSED / "parts" / f"scores_{fs}_fold{k}.parquet", index=False)
            b.to_json(PROCESSED / "parts" / f"best_{fs}_fold{k}.json", orient="records")
        raise SystemExit(0)
    scores_u, best_u = run("unaware", FEATURES_UNAWARE, df, folds)
    scores_a, best_a = run("aware", FEATURES_AWARE, df, folds)
    scores = pd.concat([scores_u, scores_a], ignore_index=True)
    scores.to_parquet(PROCESSED / "scores.parquet", index=False)
    best = pd.concat([best_u, best_a], ignore_index=True)
    best.to_csv(TABLES / "rq1_xgb_best_params.csv", index=False)
    best.to_json(PROCESSED / "xgb_best_params.json", orient="records")
    print(scores.groupby(["feature_set", "model", "partition"]).size())

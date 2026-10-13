"""Recode the raw file, build derived variables, and assign outer folds and
fit/calibration partitions. Writes data/processed/clients_clean.parquet and
data/processed/folds.csv. Every random process uses seed 640."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from common import (N_FOLDS, PAY_COLS, PROCESSED, RAW, SEED, TARGET)


def recode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": TARGET}).copy()
    # EDUCATION: documented 1..4; undocumented 0, 5, 6 kept as one distinct level, coded 0.
    df["EDUCATION"] = df["EDUCATION"].replace({5: 0, 6: 0})
    # MARRIAGE: documented 1..3; undocumented 0 stays as its own level.
    # AGE_BAND for RQ2 (not a model feature).
    df["AGE_BAND"] = pd.cut(df["AGE"], bins=[0, 29, 39, 49, 200],
                            labels=["<30", "30-39", "40-49", ">=50"]).astype(str)
    # Engineered features.
    df["UTILISATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"]
    denom = df["BILL_AMT2"]
    ratio = np.where(denom > 0, df["PAY_AMT1"] / denom.where(denom > 0, 1), 1.0)
    df["PAY_RATIO"] = np.minimum(ratio, 1.0)
    df["MAX_DELAY"] = df[PAY_COLS].max(axis=1)
    return df


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    strata = df["SEX"].astype(str) + "_" + df[TARGET].astype(str)
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = pd.DataFrame({"ID": df["ID"].values, "outer_fold": -1})
    parts = {f"part_fold{k}": np.array(["test"] * len(df), dtype=object) for k in range(N_FOLDS)}
    for k, (train_idx, test_idx) in enumerate(outer.split(df, strata)):
        folds.loc[test_idx, "outer_fold"] = k
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + k)
        fit_rel, cal_rel = next(sss.split(train_idx, strata.iloc[train_idx]))
        col = parts[f"part_fold{k}"]
        col[train_idx[fit_rel]] = "fit"
        col[train_idx[cal_rel]] = "cal"
    for k, v in parts.items():
        folds[k] = v
    return folds


if __name__ == "__main__":
    raw = pd.read_excel(RAW, header=1)
    clean = recode(raw)
    folds = assign_folds(clean)
    clean.to_parquet(PROCESSED / "clients_clean.parquet", index=False)
    folds.to_csv(PROCESSED / "folds.csv", index=False)
    print(clean.shape, folds["outer_fold"].value_counts().sort_index().to_dict())
    for k in range(N_FOLDS):
        vc = folds[f"part_fold{k}"].value_counts().to_dict()
        print(f"fold {k}:", vc)
    print("EDUCATION levels:", clean["EDUCATION"].value_counts().sort_index().to_dict())
    print("MARRIAGE levels:", clean["MARRIAGE"].value_counts().sort_index().to_dict())
    print("AGE_BAND:", clean["AGE_BAND"].value_counts().to_dict())

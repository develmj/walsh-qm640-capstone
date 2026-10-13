"""Fetch the UCI Default of Credit Card Clients file and verify its checksum.

Primary source: UCI Machine Learning Repository, dataset 350,
https://doi.org/10.24432/C55S3H (Yeh, 2009). A public GitHub mirror of the
identical XLS is used as a fallback when the UCI host is unreachable. The
SHA-256 below was recorded from the file used in this study; the structural
checks confirm the documented row count, column set, class balance, and
group counts whichever source served the bytes.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "default of credit card clients.xls"
SHA256 = "c76095b2a65c6e1b0ef6bdfab785346065b02f3f50cab6751d5a46b4c01c9e44"
SOURCES = [
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/master/dataset/default%20of%20credit%20card%20clients.xls",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> Path:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        return DEST
    for url in SOURCES:
        try:
            print(f"fetching {url}")
            data = urllib.request.urlopen(url, timeout=60).read()
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")
            continue
        if url.endswith(".zip"):
            import io
            import zipfile
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".xls")][0]
                data = z.read(name)
        DEST.write_bytes(data)
        return DEST
    sys.exit("no source reachable; place the UCI XLS at data/raw/ manually")


def verify(path: Path) -> None:
    digest = sha256(path)
    print(f"sha256 {digest}")
    if digest != SHA256:
        print("WARNING: checksum differs from the recorded value; structural checks follow")
    df = pd.read_excel(path, header=1)
    assert df.shape == (30000, 25), df.shape
    y = df["default payment next month"]
    assert int(y.sum()) == 6636, int(y.sum())
    assert df["SEX"].value_counts().to_dict() == {2: 18112, 1: 11888}
    assert df["ID"].is_unique and df.isna().sum().sum() == 0
    print("structure verified: 30,000 x 25, 6,636 defaults, 11,888 male / 18,112 female")


if __name__ == "__main__":
    verify(fetch())

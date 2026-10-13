# QM640 capstone pipeline. Run `make all` from a Python 3.11+ environment with
# `pip install -r requirements.txt` completed. Every random process uses seed 640.
PY ?= python3
XGB_ITER ?= 12

.PHONY: all data train rq1 rq2 rq3 rq4 sensitivity extensions register sample-size clean

all: data rq2 train rq1 rq3 rq4 sensitivity extensions register sample-size

data:
	$(PY) src/download.py
	$(PY) src/prepare.py

rq2:
	$(PY) analysis/01_eda.py
	$(PY) analysis/02_rq2_outcome_disparity.py

train:
	XGB_ITER=$(XGB_ITER) $(PY) src/train.py

rq1:
	$(PY) analysis/03_rq1_models.py

rq3:
	$(PY) analysis/04_rq3_fairness_audit.py

rq4:
	$(PY) analysis/05_rq4_mitigation.py

sensitivity:
	$(PY) analysis/06_sensitivity_shap.py

extensions:
	$(PY) analysis/07_rq4_extensions.py

register:
	$(PY) analysis/08_statistical_depth.py

sample-size:
	$(PY) src/sample_size.py

clean:
	rm -f data/processed/*.parquet data/processed/*.csv data/processed/*.json results/tables/*.csv results/figures/*.png

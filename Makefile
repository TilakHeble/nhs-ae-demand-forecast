PY ?= python

.PHONY: all data backtest forecast calibrate four_hour notebooks test lint app

all: data backtest forecast calibrate four_hour notebooks

data:            ## raw NHS England files -> tidy trust-month panel + hierarchy (data contracts run here)
	$(PY) -m nhsae.panel

backtest:        ## rolling-origin backtest, both measures, both COVID treatments (~25 min on 2 cores)
	$(PY) -m nhsae.backtest --measure attendances --covid impute
	$(PY) -m nhsae.backtest --measure attendances --covid post
	$(PY) -m nhsae.backtest --measure emergency_admissions --covid impute
	$(PY) -m nhsae.backtest --measure emergency_admissions --covid post
	$(PY) -m nhsae.report

forecast:        ## refit the winning configuration on all data, 12-month coherent forecasts
	$(PY) -m nhsae.forecast --measure attendances
	$(PY) -m nhsae.forecast --measure emergency_admissions

calibrate:       ## choose and fit interval calibration (online conformal test), then refresh forecasts
	$(PY) -m nhsae.calibrate
	$(MAKE) forecast

four_hour:       ## 4-hour performance risk: quantile LightGBM backtest + live forecast (~2 min)
	$(PY) -m nhsae.four_hour

notebooks:       ## execute notebooks/src/*.py into notebooks/*.ipynb and write docs/figures
	$(PY) scripts/build_notebooks.py

test:
	pytest -q

lint:
	ruff check src tests app scripts notebooks/src

app:
	streamlit run app/streamlit_app.py

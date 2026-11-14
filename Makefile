PY ?= python

.PHONY: all data backtest forecast app

all: data backtest forecast

data:            ## raw NHS England files -> tidy trust-month panel + hierarchy
	$(PY) src/build_panel.py

backtest:        ## rolling-origin backtest, both measures, both COVID treatments (~25 min on 2 cores)
	$(PY) src/backtest.py --measure attendances --covid impute
	$(PY) src/backtest.py --measure attendances --covid post
	$(PY) src/backtest.py --measure emergency_admissions --covid impute
	$(PY) src/backtest.py --measure emergency_admissions --covid post
	$(PY) src/report.py

forecast:        ## refit the winning configuration on all data, 12-month coherent forecasts
	$(PY) src/forecast.py --measure attendances
	$(PY) src/forecast.py --measure emergency_admissions

app:
	streamlit run app/streamlit_app.py

# Glossary

**A&E types.** *Type 1*: consultant-led 24-hour major emergency department. *Type 2*: single-specialty (for example eye, dental). *Type 3/4 ("Other")*: urgent treatment centres, minor injury units, walk-in centres.

**ACF / PACF.** Autocorrelation function (correlation of a series with its own lags) and partial autocorrelation (the same, controlling for shorter lags). Used to identify ARIMA orders.

**Airline model.** ARIMA(0,1,1)(0,1,1)₁₂ on the log series (Box & Jenkins).

**Backtest / rolling origin.** Repeatedly train up to a cut-off ("origin") and forecast forward; average the errors across origins.

**Coherent.** Forecasts that respect the aggregation constraints (trusts add up to ICBs, ICBs to regions, regions to England).

**Conformal prediction.** Distribution-free calibration of intervals using the empirical quantiles of past errors. *Split conformal* uses a calibration set. *Conformalised quantile regression (CQR)* shifts predicted quantiles by their own residual quantiles.

**COVID treatment.** How the April 2020 – June 2021 break is handled: *impute* (replace with a counterfactual) or *post* (train on July 2021 onwards).

**CRPS.** Continuous ranked probability score: a proper score for a whole predictive distribution, in the units of the data.

**Damped trend.** Exponential smoothing trend multiplied by φ^h (0 < φ < 1), so long-horizon forecasts flatten out.

**Data contract.** Automated checks the data must pass before modelling (no missing months, no impossible counts, no unmapped step changes).

**Direct vs recursive forecasting.** Direct predicts y_{t+h} from information at t; recursive feeds one-step predictions back in as inputs.

**DTA.** Decision to admit. Waits from DTA to admission over 4 and 12 hours are reported separately.

**ETS.** Error-Trend-Seasonal state-space framework for exponential smoothing.

**Global model.** One model trained across many series using scale-free features.

**ICB.** Integrated Care Board: the NHS body that plans and pays for care in an area (36 in this data).

**MASE.** Mean absolute scaled error: MAE divided by the in-sample seasonal-naive MAE.

**MinT.** Minimum-trace reconciliation (Wickramasuriya, Athanasopoulos & Hyndman, 2019): the GLS projection that minimises the total variance of reconciled forecast errors.

**Pinball loss.** Quantile loss. Minimised by the true a-quantile.

**Recording change (reclassification).** A trust moving attendances between Type 1 and "Other" in its returns, usually when a co-located urgent treatment centre opens or closes. It looks like a demand shock in the Type 1 series, but it isn't one.

**Restatement ("as-is").** Re-expressing history on today's organisational structure and recording basis so that series are continuous.

**Seasonal naive.** Forecast = the same month last year.

**Shrinkage covariance.** A covariance estimate pulled towards its diagonal (Schäfer–Strimmer) so it stays well-conditioned with few observations.

**STL.** Seasonal-Trend decomposition using Loess.

**Summing matrix (S).** Maps bottom-level series to every series in the hierarchy: y = S b.

**4-hour standard.** Share of A&E attendances admitted, transferred or discharged within 4 hours of arrival. The constitutional standard is 95%; the 2026/27 objective is 82% (all types).

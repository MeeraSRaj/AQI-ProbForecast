# Probabilistic AQI Forecasting


## The idea

A point forecast says `PM2.5 = 118 µg/m³`. That is not very actionable.

A probabilistic forecast says:

```
50% interval:  105 – 130  µg/m³
90% interval:   80 – 160  µg/m³
```

Now a decision-maker knows whether tomorrow's air is likely to cross the **Poor** threshold (120 µg/m³) — and how likely. That is the difference between a statistical exercise and a useful tool.

---

## Dataset

**Air Quality Data in India (2015–2020)** — CPCB via Kaggle  
Hourly readings · Multiple cities · Target feature: PM2.5 (µg/m³)

AQI categories used as threshold lines in all forecast plots:

| Category     | PM2.5 (µg/m³) |
|-------------|--------------|
| Good         | 0 – 30       |
| Satisfactory | 30 – 60      |
| Moderate     | 60 – 90      |
| Poor         | 90 – 120     |
| Very Poor    | 120 – 250    |
| Severe       | > 250        |

---

## Models

| Model | Type | Output |
|---|---|---|
| ARIMA | Classical stats | Point forecast |
| Prophet | Additive decomposition | Point forecast |
| LSTM | Deep learning | Point forecast |
| Quantile LSTM | Deep learning + pinball loss | 10th / 50th / 90th percentile |

The quantile model is the centrepiece — the only one that produces a confidence band rather than a line.

---

## Metrics

Standard models are evaluated on RMSE and MAE.  
The probabilistic model is additionally evaluated on:

- **Pinball Loss** — penalises quantile over/under-shooting
- **PICP** (Prediction Interval Coverage Probability) — what fraction of actuals fall inside the predicted band?
- **CRPS** (Continuous Ranked Probability Score) — overall sharpness + calibration in one number

---

## Tech stack

Python · PyTorch · statsmodels · Prophet · pandas · matplotlib

---

*3-week portfolio project · Probabilistic time series · India AQI*
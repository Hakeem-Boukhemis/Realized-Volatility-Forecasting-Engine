# Realized Volatility Forecasting Comparison

A replication and benchmark of the **Heterogeneous Autoregressive model of Realized Volatility (HAR-RV)** from Corsi (2009), evaluated against a Random Forest and a shallow MLP on S&P 500 realized volatility. Performance is assessed using QLIKE loss, Mincer-Zarnowitz regressions, and Diebold-Mariano tests against a naïve persistence baseline.

---

## Motivation

Corsi's HAR-RV remains one of the most cited volatility forecasting models in the empirical finance literature — a linear model that captures long-memory behavior in realized volatility through a cascade of heterogeneous trading horizons, without the estimation complexity of a true long-memory process. This project asks whether a Random Forest or a shallow neural network can extract additional signal from the same feature set, and whether any of the three beats a naïve benchmark under a realistic, asymmetric loss function.

---

## Data

**Oxford-Man Realized Library** — publicly available dataset of daily realized measures for major equity indices, constructed from high-frequency intraday data.

- **Symbol:** `.SPX` (S&P 500)
- **Realized measure:** `rv5` — realized variance computed from 5-minute intraday returns
- **Train / test split:** 80 / 20, chronological (no shuffling)
- **Test period:** approximately 2014–2018

The dataset is no longer available through Oxford-Man, however it is included in this repository under  `data/oxfordmanrealizedvolatilityindices.csv`.

---

## Model

### HAR-RV (Corsi 2009)

The model decomposes realized volatility into three heterogeneous components corresponding to daily, weekly, and monthly trader horizons:

$$\log RV_{t+1} = \beta_0 + \beta_d \log RV_t^{(d)} + \beta_w \log RV_t^{(w)} + \beta_m \log RV_t^{(m)} + \varepsilon_{t+1}$$

where:

| Component | Definition |
|---|---|
| $RV_t^{(d)}$ | $\log RV_t$ — previous day |
| $RV_t^{(w)}$ | $\frac{1}{5}\sum_{i=0}^{4} \log RV_{t-i}$ — 5-day average |
| $RV_t^{(m)}$ | $\frac{1}{22}\sum_{i=0}^{21} \log RV_{t-i}$ — 22-day average |

The model is estimated via OLS with HAC standard errors (Newey-West, 5 lags) to account for serial correlation in residuals.

### Random Forest

A 100-estimator Random Forest using the same three HAR features (`RV_daily`, `RV_weekly`, `RV_monthly`) as inputs. No constant term. Included to test whether a non-linear ensemble extracts additional signal from the same feature set.

### MLP

A shallow two-layer network trained on the same HAR features for 200 epochs using Adam and MSE loss. Features and target are scaled independently; predictions are inverse-transformed before evaluation.

---

## Evaluation

### QLIKE Loss

$$\mathcal{L}_{QLIKE} = \frac{\sigma^2}{\hat{\sigma}^2} - \log\frac{\sigma^2}{\hat{\sigma}^2} - 1$$

QLIKE is the standard loss function for volatility forecast evaluation (Patton, 2011). It is asymmetric — over-prediction is penalized more heavily than under-prediction of equal magnitude. This makes it a stricter and more realistic benchmark than MSE.

### Mincer-Zarnowitz Regression

Tests forecast efficiency via the regression $\sigma^2_t = \alpha + \beta\hat{\sigma}^2_t + \varepsilon_t$. An efficient forecast has $\alpha = 0$ and $\beta = 1$, evaluated via a joint F-test. Estimated with HAC standard errors.

### Diebold-Mariano Test

Tests whether the difference in forecast accuracy between two models is statistically significant, using a Newey-West adjusted variance estimator for the loss differential. $H_0$: equal forecast accuracy.

---

## Results

All three models — HAR-RV, Random Forest, and MLP — are outperformed by the naïve persistence benchmark on QLIKE loss over the test period.

**This is an expected and documented result, not an error.** Several factors explain it:

- **QLIKE asymmetry:** HAR's smoothed weekly and monthly components mean-revert slowly. In low-volatility regimes, they cause the model to systematically over-predict realized variance, which QLIKE penalizes disproportionately. Naïve persistence adapts in one step.
- **Test regime:** The 2014–2018 test window was a prolonged low-volatility environment (with isolated spikes). This is precisely the regime where HAR's long-horizon components are a liability rather than an asset.
- **Feature ceiling for ML models:** The Random Forest and MLP use the same three HAR components as their input. Without additional features, they cannot obtain information that the linear model doesn't already capture.
                  QLIKE
Model                  
Naive          0.292132
HAR-RV         0.402441
Random Forest  0.477593
MLP            0.389185


---

## Limitations & Next Steps (V2)

- **Static train/test split:** A walk-forward (rolling window) backtest would give a more robust picture of out-of-sample performance across multiple regimes. This is the primary planned extension.
- **Narrow feature set for ML models:** RF and MLP use only three features, not allowing them to have enough predictors. Adding exogenous regressors (e.g., VIX, GEX, macro variables) would constitute a more meaningful ML benchmark. 
- **MLP architecture:** A single hidden layer trained with MSE is a relatively weak baseline. A HAR-GARCH hybrid or a sequence model (LSTM) would be a more appropriate comparison.


## Dependencies

```
numpy
pandas
statsmodels
scikit-learn
torch
scipy
matplotlib
```

Install with:
```bash
pip install numpy pandas statsmodels scikit-learn torch scipy matplotlib
```

---

## References

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics.



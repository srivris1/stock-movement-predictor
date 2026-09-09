import json
import os


def md(source: str) -> dict:
    """Create a markdown cell."""
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(source: str) -> dict:
    """Create a code cell."""
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


cells = []

cells.append(md("""# Stock Price Movement Predictor

**Track**: Time-Series Machine Learning
**Tech Stack**: Python, pandas, scikit-learn, matplotlib
**Author**: Rishit Srivastava
**Objective**: Predict next-day price direction (Up/Down) for a liquid equity using
engineered temporal features, with rigorous leak-free methodology, dual naive baselines,
forward time-series splits, walk-forward cross-validation, economic backtesting, and
SHAP-based model explainability.

---"""))

cells.append(md("## 0. Setup & Imports"))
cells.append(code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg

np.random.seed(cfg.RANDOM_SEED)
print(f"Ticker: {cfg.TICKER}")
print(f"Seed:   {cfg.RANDOM_SEED}")
print(f"Train:  {cfg.DATA_START} to {cfg.TRAIN_TEST_SPLIT_DATE}")
print(f"Test:   {cfg.TRAIN_TEST_SPLIT_DATE} to {cfg.DATA_END}")"""))

cells.append(md("## 1. Data Loading\n\nDownload daily OHLCV data from Yahoo Finance via `yfinance`. The result is cached as a CSV in `data/` so subsequent runs don't re-download."))
cells.append(code("""from src.data_loader import download_ohlcv

df = download_ohlcv()
print(f"Shape: {df.shape}")
df.head(10)"""))

cells.append(md("""## 2. Leak-Free Target Construction

The target label for day *t* is:

```
target[t] = 1  if  Close[t+1] > Close[t]    (price goes UP tomorrow)
target[t] = 0  otherwise                     (price goes DOWN tomorrow)
```

This is a **future** label — it tells us what will happen *after* day t.
The key constraint is that **no feature** for day t may use any information from day t+1 or later.
The last row is dropped because its future close is unknown."""))

cells.append(code("""from src.data_loader import construct_target, verify_no_leakage

df = construct_target(df)
verify_df = verify_no_leakage(df, n_rows=10)
verify_df"""))

cells.append(md("""## 3. Feature Engineering

### 3a. Raw Price/Volume Features

Simple features derived directly from OHLCV data — no look-ahead bias possible
since each feature uses only data from day t and earlier (lag features use `shift()`).

### 3b. Engineered Technical Indicators

Seven technical indicators computed in pure pandas:
- **RSI (14)**: Relative Strength Index
- **MACD (12, 26, 9)**: Moving Average Convergence Divergence (line, signal, histogram)
- **Bollinger %B (20, 2)**: Position within Bollinger Bands
- **ATR (14)**: Average True Range, normalized by close
- **EMA Ratios**: Close / EMA for 10, 20, and 50-period windows
- **Volume Ratio**: Volume / 20-day SMA of volume
- **Stochastic %K (14)**: Stochastic oscillator
- **OBV Change**: On-Balance Volume 20-day rate of change

Every indicator is backward-looking by construction — it only aggregates past and present data."""))

cells.append(code("""from src.features import (
    add_raw_features, add_technical_indicators,
    get_feature_matrix, RAW_FEATURE_COLS, ENGINEERED_FEATURE_COLS, ALL_FEATURE_COLS,
)

df = add_raw_features(df)
df = add_technical_indicators(df)
print(f"Total features: {len(ALL_FEATURE_COLS)}")
print(f"Raw features:   {len(RAW_FEATURE_COLS)}")
print(f"Engineered:     {len(ENGINEERED_FEATURE_COLS)}")
df[ALL_FEATURE_COLS].describe().T"""))

cells.append(md("### Price Colored by Next-Day Direction"))
cells.append(code("""from src.visualize import plot_price_with_direction
fig, ax = plot_price_with_direction(df)
plt.show()"""))

cells.append(md("""## 4. Time-Based Train/Test Split

Everything strictly **before** the split date goes to training; everything **on or after** goes to testing.
This prevents any temporal leakage from test data into the training process.
The StandardScaler will be fitted on the training partition ONLY."""))

cells.append(code("""from src.data_loader import split_by_date

train_df, test_df = split_by_date(df)"""))

cells.append(md("## 5. Class Balance Report"))
cells.append(code("""from src.visualize import plot_class_balance

X_train_raw, y_train = get_feature_matrix(train_df, RAW_FEATURE_COLS)
X_test_raw, y_test = get_feature_matrix(test_df, RAW_FEATURE_COLS)

print("Train distribution:")
print(y_train.value_counts())
print(f"\\nTest distribution:")
print(y_test.value_counts())

fig, axes = plot_class_balance(y_train, y_test)
plt.show()"""))

cells.append(md("""## 6. Model Training – Raw Features

Three classifiers (Logistic Regression, Random Forest, Gradient Boosting) trained
on raw price/volume features. The scaler is fitted on training data ONLY, then
applied to test data.

We also compute two naive baselines:
- **Persistence**: Predict that tomorrow's direction equals today's direction.
- **Majority Class**: Always predict the most common class from training."""))

cells.append(code("""from src.model import train_and_evaluate

raw_results, raw_preds = train_and_evaluate(
    X_train_raw, y_train, X_test_raw, y_test, feature_set_name="Raw"
)
raw_results"""))

cells.append(md("## 7. Model Training – Engineered Features\n\nSame three classifiers, now trained on the full feature set (raw + 12 technical indicators)."))
cells.append(code("""X_train_eng, y_train_eng = get_feature_matrix(train_df, ALL_FEATURE_COLS)
X_test_eng, y_test_eng = get_feature_matrix(test_df, ALL_FEATURE_COLS)

eng_results, eng_preds = train_and_evaluate(
    X_train_eng, y_train_eng, X_test_eng, y_test_eng, feature_set_name="Engineered"
)
eng_results"""))

cells.append(md("""## 8. Four-Way Comparison Table

**Persistence vs Majority Class vs Raw Features vs Engineered Features**

This is the core deliverable: a direct comparison showing whether engineered
temporal features provide any lift over naive baselines and raw features."""))

cells.append(code("""four_way = pd.concat([raw_results, eng_results.iloc[2:]], ignore_index=True)
print(four_way.to_string(index=False))

from src.visualize import plot_comparison_table
fig, ax = plot_comparison_table(four_way, title="Four-Way Comparison: Baselines vs Raw vs Engineered")
plt.show()"""))

cells.append(md("## 9. Prediction vs Actual Direction Plot"))
cells.append(code("""from src.visualize import plot_prediction_vs_actual, plot_confusion_matrices
from src.model import get_classification_report

# Find best model
best_row = four_way.iloc[2:].sort_values("Accuracy", ascending=False).iloc[0]
best_name = best_row["Model"]
print(f"Best model: {best_name} (Accuracy={best_row['Accuracy']:.4f})")

if best_name in eng_preds:
    best_preds = eng_preds[best_name]
    y_test_best = y_test_eng
    dates_best = X_test_eng.index
else:
    best_preds = raw_preds[best_name]
    y_test_best = y_test
    dates_best = X_test_raw.index

fig, axes = plot_prediction_vs_actual(dates_best, y_test_best.values, best_preds, model_name=best_name)
plt.show()

print("\\nClassification Report:")
print(get_classification_report(y_test_best, best_preds))"""))

cells.append(md("### Confusion Matrices"))
cells.append(code("""all_preds = {}
all_preds.update(raw_preds)
all_preds.update(eng_preds)

fig, axes = plot_confusion_matrices(y_test_best, all_preds)
plt.show()"""))

cells.append(md("""## 10. Walk-Forward Cross-Validation (Expanding Window)

Instead of a single static train/test split, we use an **expanding-window walk-forward validation**:
1. Start with a minimum training window of 252 trading days (~1 year).
2. Predict the next 63 trading days (~1 quarter).
3. Expand the training window by 63 days and repeat.

The scaler is **refitted from scratch** in every fold to prevent information leakage.
This provides a more robust estimate of out-of-sample performance across different
market regimes."""))

cells.append(code("""from src.model import walk_forward_cv
from src.visualize import plot_walk_forward

X_all, y_all = get_feature_matrix(df, ALL_FEATURE_COLS)
wf_results = walk_forward_cv(X_all, y_all)
print(wf_results.to_string(index=False))

fig, ax = plot_walk_forward(wf_results)
plt.show()"""))

cells.append(md("""## 11. Economic Backtest – Strategy vs Buy & Hold

Translating predictions into a simple long/cash strategy:
- When model predicts **UP** -> hold the asset (earn the daily return)
- When model predicts **DOWN** -> hold cash (earn 0)

We compare against passive buy-and-hold and compute:
- **Cumulative Return**
- **Annualized Sharpe Ratio**
- **Maximum Drawdown**
- **Win Rate** (on days we traded)"""))

cells.append(code("""from src.backtest import run_backtest, backtest_summary_table
from src.visualize import plot_backtest

test_returns = test_df["Close"].pct_change().dropna()
common_idx = dates_best.intersection(test_returns.index)
aligned_returns = test_returns.loc[common_idx]
aligned_preds = pd.Series(best_preds, index=dates_best).loc[common_idx].values

bt = run_backtest(aligned_returns, aligned_preds)
bt_table = backtest_summary_table(bt)
print(bt_table.to_string(index=False))

fig, ax = plot_backtest(bt, model_name=best_name)
plt.show()"""))

cells.append(md("""## 12. SHAP Explainability

Using TreeExplainer from the SHAP library to understand **why** the model
makes specific directional calls. This goes beyond feature importance by showing
the magnitude and direction of each feature's impact on individual predictions."""))

cells.append(code("""import shap
from sklearn.ensemble import GradientBoostingClassifier
from src.model import scale_features
from src.visualize import plot_shap_summary, plot_feature_importance

X_tr_s, X_te_s, scaler = scale_features(X_train_eng, X_test_eng)
gb_model = GradientBoostingClassifier(**cfg.GRADIENT_BOOSTING_PARAMS)
gb_model.fit(X_tr_s, y_train_eng.values)

explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_te_s)

fig, axes = plot_shap_summary(shap_values, ALL_FEATURE_COLS, X_te_s)
plt.show()"""))

cells.append(md("### Gini Feature Importance"))
cells.append(code("""fig, ax = plot_feature_importance(gb_model, ALL_FEATURE_COLS)
plt.show()"""))

cells.append(md("""## Summary

This notebook demonstrated a complete, leak-free stock movement prediction pipeline:

1. **Data**: Downloaded and cached OHLCV data from Yahoo Finance.
2. **Target**: Constructed a next-day directional label with explicit verification of zero forward leakage.
3. **Features**: Compared 13 raw price/volume features against 25 total features (raw + 12 engineered technical indicators).
4. **Methodology**: Strict time-based split with scaler fitted only on training data.
5. **Baselines**: Persistence and Majority Class baselines provide honest benchmarks.
6. **Models**: Logistic Regression, Random Forest, and Gradient Boosting evaluated with Accuracy, Precision, Recall, F1, and ROC-AUC.
7. **Walk-Forward CV**: Expanding-window validation across multiple market regimes for robust performance estimation.
8. **Economic Backtest**: Simulated long/cash strategy with Sharpe Ratio, Max Drawdown, and Win Rate vs Buy & Hold.
9. **SHAP Explainability**: Feature attribution analysis explaining individual model decisions.

All configurations are centralized in `config.py` with pinned random seeds and hyperparameters for exact reproducibility."""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "analysis_notebook.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {os.path.abspath(out_path)}")

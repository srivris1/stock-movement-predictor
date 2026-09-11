# Stock Movement Predictor

This is a machine learning project that tries to predict whether a stocks closing price will go up or down the next trading day. I used Apple (AAPL) as the default ticker but you can change it to any stock or index in config.py.

The whole point was to build a proper end-to-end ML pipeline, not just throw data at a model and hope for the best. I wanted to see if adding technical indicators like RSI and MACD actually helps compared to just using raw price data, and whether any model can consistently beat two really simple baselines (just guessing the majority class, or assuming tomorrow will be the same as today).

Spoiler: its harder than you think. Markets are noisy and daily direction is a brutal target.

## How It Works

1. Download daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance using yfinance
2. Build a binary target label: did the price go UP the next day, yes or no
3. Extract 13 raw features from the price data (returns, range, gaps, lags)
4. Add 12 engineered technical indicators on top (RSI, MACD, Bollinger Bands, etc)
5. Split the data by date so the model only trains on past data and tests on future data
6. Train three classifiers and compare them against two naive baselines
7. Generate a four-way comparison table showing which approach works best
8. Run walk-forward cross validation to see if results hold across different time periods
9. Simulate a trading strategy to see if the predictions actually make money
10. Use SHAP to explain why the model makes the predictions it does

## Tech Stack

- **Python** - main language for everything
- **pandas** - data manipulation, feature engineering, all the indicator math
- **scikit-learn** - StandardScaler for normalization, LogisticRegression, RandomForestClassifier, GradientBoostingClassifier for modeling, plus all the evaluation metrics
- **matplotlib** - all 9 figures (price plots, comparison tables, confusion matrices, equity curves, etc)
- **seaborn** - some of the plot styling
- **yfinance** - pulling historical stock data from Yahoo Finance
- **ta** - cross checking my pandas indicator implementations
- **shap** - TreeExplainer for understanding feature attributions
- **nbformat** - generating the Jupyter notebook programmatically

## Target Construction (No Leakage)

This was the most important part to get right. The label for day t is:

target[t] = 1 if Close[t+1] > Close[t], else 0

So we are predicting whether tomorrows close will be higher than todays close. The tricky part is making sure none of the features accidentally use future information. Every feature for day t only uses data from day t and earlier, nothing from t+1 or beyond.

I added a verification step that prints the features alongside the target and the actual next-day close so you can manually check that nothing is leaking. The notebook prints this table and confirms zero mismatches.

The last row of the dataset gets dropped since we dont know what happens after it.

## Technical Indicators

All of these are computed in plain pandas. Each one only looks at past and present data:

- RSI (14 period) - measures if a stock is overbought or oversold
- MACD (12, 26, 9) - trend following indicator, I use the line, signal, and histogram
- Bollinger %B (20 period, 2 std) - where the price sits relative to its Bollinger Bands
- ATR (14 period) - average true range, normalized by close, measures volatility
- EMA ratios - close price divided by 10, 20, and 50 day exponential moving averages
- Volume ratio - todays volume compared to its 20-day average
- Stochastic %K (14 period) - momentum oscillator
- OBV rate of change - on balance volume trend over 20 days

Thats 12 engineered features on top of 13 raw features (daily return, intraday range, body ratio, gap, volume change, and lag features for the past 1/2/3/5 days). Total of 25 features.

## Models and Baselines

I trained three scikit-learn classifiers:
- Logistic Regression (simple linear baseline)
- Random Forest (200 trees, max depth 8)
- Gradient Boosting (200 estimators, learning rate 0.05)

And compared them against two dumb baselines:
- **Persistence** - predict that tomorrow will be the same direction as today
- **Majority Class** - always predict UP since thats the slightly more common class (~54%)

The StandardScaler is fitted ONLY on the training data and then applied to test. This is important because fitting on the full dataset would leak test information into training. All random seeds are pinned to 42 in config.py.

## What Stands Out

These are the things I did beyond the basic requirements that I think make this project different:

**Walk-Forward Cross Validation** - Instead of just one train/test split I used an expanding window approach across 26 quarterly folds. The model trains on increasingly larger windows and predicts the next quarter each time. This gives a much more honest picture of performance across different market conditions (bull runs, crashes, sideways markets). Mean accuracy was 48.9% which is more realistic than the single-split number.

**Economic Backtest** - I translated the model predictions into an actual simulated trading strategy. When the model predicts UP you hold the stock, when it predicts DOWN you hold cash. Then I compared this against just buying and holding the stock. The model returned about 25% over the test period vs 58% for buy-and-hold. But the model had way less drawdown (-11% vs much worse). I also calculated the Sharpe ratio (0.79) and win rate (53%) to give a full picture.

**SHAP Explainability** - I used TreeExplainer from the SHAP library to understand which features the model actually relies on. This goes beyond just looking at Gini importance because SHAP shows both the magnitude and direction of each features contribution. Turns out range_lag_5 and RSI were the most important features.

**CLI Prediction Tool** - Theres a standalone script (predict.py) that lets you get quick directional predictions for any ticker from the command line. Just run `python predict.py --ticker MSFT --days 20` and it downloads fresh data, trains a model, and shows predictions with confidence scores.

## Results

Best model was Random Forest with the full feature set at around 55% directional accuracy. Not amazing but thats about right for daily direction prediction. Anyone claiming 70%+ is almost certainly leaking data or overfitting.

The walk-forward CV gave 48.9% mean accuracy across 26 folds which is more honest. Some folds hit 63%, some dropped to 38%. Markets are just noisy at the daily level.

The backtest showed 25% return for the model strategy vs 58% for buy-and-hold. The model was more conservative with only -11% max drawdown. Sharpe ratio was 0.79.

Honest takeaway: the model picks up something real, it does beat both baselines, but daily direction prediction with technical indicators alone is extremely hard.

## Files

- `config.py` - all settings, hyperparameters, dates, random seed (42)
- `main.py` - runs the entire 10-step pipeline end to end
- `predict.py` - CLI tool for quick predictions on any ticker
- `create_notebook.py` - generates the Jupyter notebook
- `analysis_notebook.ipynb` - the main deliverable notebook with full analysis
- `requirements.txt` - exact pinned package versions for reproducibility
- `src/data_loader.py` - downloads OHLCV data, caches it locally, builds the target, splits train/test by date
- `src/features.py` - raw feature extraction and all 12 technical indicator implementations
- `src/model.py` - model training, evaluation metrics, persistence and majority baselines, walk-forward CV
- `src/backtest.py` - trading strategy simulation, Sharpe ratio, drawdown, win rate
- `src/visualize.py` - generates all 9 matplotlib figures
- `figures/` - the 9 saved plots (price direction, class balance, comparison table, predictions, confusion matrices, walk-forward, backtest equity curve, SHAP summary, feature importance)
- `data/` - cached OHLCV csv and output tables

## How To Run

Install the dependencies (all versions are pinned):
```
pip install -r requirements.txt
```

Run the full pipeline:
```
python main.py
```
This downloads data, trains models, generates all plots, and saves everything. Takes about 30 seconds.

Open the notebook:
```
jupyter notebook analysis_notebook.ipynb
```

Quick predictions:
```
python predict.py --ticker AAPL --days 20
```

## Reproducibility

Everything is pinned for exact reproducibility:
- Python 3.13
- All package versions locked in requirements.txt
- Random seed = 42 in config.py
- All model hyperparameters centralized in config.py
- Data gets cached as CSV after first download

Running `python main.py` twice with the same config gives identical results.

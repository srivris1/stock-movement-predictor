import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

import config as cfg
from src.data_loader import download_ohlcv, construct_target
from src.features import add_raw_features, add_technical_indicators, ALL_FEATURE_COLS, get_feature_matrix


def predict(ticker: str, days: int):
    np.random.seed(cfg.RANDOM_SEED)

    print(f"\n  Downloading latest data for {ticker} ...")
    df = download_ohlcv(ticker=ticker)
    df = construct_target(df)
    df = add_raw_features(df)
    df = add_technical_indicators(df)

    X, y = get_feature_matrix(df, ALL_FEATURE_COLS)

    split = max(len(X) - days, int(len(X) * 0.8))
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_recent = X.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_re_s = scaler.transform(X_recent)

    model = GradientBoostingClassifier(**cfg.GRADIENT_BOOSTING_PARAMS)
    model.fit(X_tr_s, y_train.values)

    preds = model.predict(X_re_s)
    proba = model.predict_proba(X_re_s)[:, 1]

    print(f"\n  Predictions for {ticker} (last {len(X_recent)} trading days):\n")
    print(f"  {'Date':<12}  {'Predicted':<10}  {'Confidence':<12}  {'Actual':<8}")
    print(f"  {'-'*46}")

    actual = y.iloc[split:]
    for i, (date, pred, prob) in enumerate(zip(X_recent.index, preds, proba)):
        direction = "UP" if pred == 1 else "DOWN"
        act = "UP" if actual.iloc[i] == 1 else "DOWN" if i < len(actual) else "?"
        conf = f"{prob:.1%}" if pred == 1 else f"{1-prob:.1%}"
        marker = "+" if (i < len(actual) and pred == actual.iloc[i]) else "x"
        print(f"  {date.strftime('%Y-%m-%d'):<12}  {direction:<10}  {conf:<12}  {act:<8} {marker}")

    correct = (preds == actual.values).sum()
    total = len(preds)
    print(f"\n  Directional accuracy: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next-day stock direction")
    parser.add_argument("--ticker", type=str, default=cfg.TICKER, help="Ticker symbol")
    parser.add_argument("--days", type=int, default=20, help="Number of recent days to predict")
    args = parser.parse_args()
    predict(args.ticker, args.days)

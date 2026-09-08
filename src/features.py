import numpy as np
import pandas as pd
import config as cfg


def add_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive simple price-action and volume features from raw OHLCV columns.
    These do NOT use any look-ahead data.
    """
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["body_ratio"] = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-9)
    df["volume_change"] = df["Volume"].pct_change()

    for lag in [1, 2, 3, 5]:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag)
        df[f"range_lag_{lag}"] = df["intraday_range"].shift(lag)

    return df


RAW_FEATURE_COLS = [
    "daily_return", "intraday_range", "body_ratio", "gap", "volume_change",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "range_lag_1", "range_lag_2", "range_lag_3", "range_lag_5",
]


def _rsi(series: pd.Series, period: int = cfg.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index – computed in pure pandas."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = series.ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _bollinger_pct_b(series: pd.Series) -> pd.Series:
    """Bollinger %B: position of price relative to the bands."""
    sma = series.rolling(cfg.BOLLINGER_PERIOD).mean()
    std = series.rolling(cfg.BOLLINGER_PERIOD).std()
    upper = sma + cfg.BOLLINGER_STD * std
    lower = sma - cfg.BOLLINGER_STD * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return pct_b


def _atr(df: pd.DataFrame, period: int = cfg.ATR_PERIOD) -> pd.Series:
    """Average True Range (normalized by close)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr / (df["Close"] + 1e-9)


def _stochastic_k(df: pd.DataFrame, period: int = cfg.STOCHASTIC_PERIOD) -> pd.Series:
    """Stochastic %K oscillator."""
    lowest = df["Low"].rolling(period).min()
    highest = df["High"].rolling(period).max()
    return 100 * (df["Close"] - lowest) / (highest - lowest + 1e-9)


def _obv_pct_change(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """On-Balance Volume rate of change (normalized)."""
    direction = np.sign(df["Close"].diff())
    obv = (direction * df["Volume"]).cumsum()
    return obv.pct_change(period)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7+ engineered technical indicators.
    All use only data ≤ t.
    """
    df = df.copy()

    df["rsi_14"] = _rsi(df["Close"])

    macd_line, macd_signal, macd_hist = _macd(df["Close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_histogram"] = macd_hist

    df["bollinger_pct_b"] = _bollinger_pct_b(df["Close"])

    df["atr_norm"] = _atr(df)

    df["ema_ratio_10"] = df["Close"] / df["Close"].ewm(span=cfg.EMA_SHORT, adjust=False).mean()
    df["ema_ratio_20"] = df["Close"] / df["Close"].ewm(span=cfg.EMA_MEDIUM, adjust=False).mean()
    df["ema_ratio_50"] = df["Close"] / df["Close"].ewm(span=cfg.EMA_LONG, adjust=False).mean()

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(cfg.VOLUME_SMA_PERIOD).mean()

    df["stochastic_k"] = _stochastic_k(df)

    df["obv_change"] = _obv_pct_change(df)

    return df


ENGINEERED_FEATURE_COLS = [
    "rsi_14",
    "macd_line", "macd_signal", "macd_histogram",
    "bollinger_pct_b",
    "atr_norm",
    "ema_ratio_10", "ema_ratio_20", "ema_ratio_50",
    "volume_ratio",
    "stochastic_k",
    "obv_change",
]

ALL_FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_FEATURE_COLS


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) with NaN rows dropped, suitable for model training.
    """
    subset = df[feature_cols + ["target"]].dropna()
    X = subset[feature_cols]
    y = subset["target"]
    return X, y

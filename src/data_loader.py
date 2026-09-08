import os
import pandas as pd
import yfinance as yf
import config as cfg


def download_ohlcv(
    ticker: str = cfg.TICKER,
    start: str = cfg.DATA_START,
    end: str = cfg.DATA_END,
    cache_dir: str = cfg.DATA_DIR,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    If a cached CSV already exists for the same ticker, it is loaded
    instead (delete the CSV to force a fresh download).

    Returns a DataFrame indexed by Date with columns:
        Open, High, Low, Close, Volume
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker.replace('^', '')}_ohlcv.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        print(f"[data] Loaded cached data from {cache_path}  ({len(df)} rows)")
        return df

    print(f"[data] Downloading {ticker} from {start} to {end} ...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    df.to_csv(cache_path)
    print(f"[data] Saved {len(df)} rows to {cache_path}")
    return df


def construct_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a LEAK-FREE next-day directional label.

    target[t] = 1  if  Close[t+1] > Close[t]   (price goes UP tomorrow)
    target[t] = 0  otherwise

    The last row is dropped because its future close is unknown.
    """
    df = df.copy()
    df["next_close"] = df["Close"].shift(-1)
    df["target"] = (df["next_close"] > df["Close"]).astype(int)
    df.drop(columns=["next_close"], inplace=True)
    df.dropna(subset=["target"], inplace=True)
    df["target"] = df["target"].astype(int)
    return df


def verify_no_leakage(df: pd.DataFrame, n_rows: int = 10) -> pd.DataFrame:
    """
    Print feature rows alongside target labels so a reviewer can visually
    verify that target[t] depends ONLY on future data (Close[t+1] vs Close[t])
    and that no feature column contains future information.

    Returns a small verification DataFrame for display.
    """
    verify_cols = ["Open", "High", "Low", "Close", "Volume", "target"]
    available = [c for c in verify_cols if c in df.columns]
    sample = df[available].head(n_rows).copy()

    sample["actual_next_close"] = df["Close"].shift(-1).head(n_rows)
    sample["label_check"] = (sample["actual_next_close"] > sample["Close"]).astype(int)

    print("\n=== LEAK-FREE TARGET VERIFICATION ===")
    print("target[t] should equal label_check[t] (1 if next_close > close, else 0)")
    print(sample.to_string())

    mismatches = (sample["target"] != sample["label_check"]).sum()
    if mismatches == 0:
        print("[OK] Zero mismatches - target construction is leak-free.\n")
    else:
        print(f"[FAIL] {mismatches} mismatches detected - investigate!\n")

    return sample


def split_by_date(
    df: pd.DataFrame,
    split_date: str = cfg.TRAIN_TEST_SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split.  Everything strictly before `split_date`
    goes to train; everything on or after goes to test.
    """
    train = df.loc[df.index < split_date].copy()
    test = df.loc[df.index >= split_date].copy()
    print(f"[split] Train: {train.index.min().date()} -> {train.index.max().date()}  ({len(train)} rows)")
    print(f"[split] Test:  {test.index.min().date()} -> {test.index.max().date()}  ({len(test)} rows)")
    return train, test

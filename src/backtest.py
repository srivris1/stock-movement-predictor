import numpy as np
import pandas as pd
import config as cfg


def run_backtest(
    daily_returns: pd.Series,
    y_pred: np.ndarray,
    risk_free: float = cfg.RISK_FREE_RATE,
    trading_days: int = cfg.TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Simulate a long-only strategy:
      - When the model predicts UP  → hold the asset (earn daily return)
      - When the model predicts DOWN → hold cash   (earn 0)

    Parameters
    ----------
    daily_returns : pd.Series
        Actual daily log returns aligned with the test period.
    y_pred : np.ndarray
        Binary predictions (1 = UP, 0 = DOWN).

    Returns
    -------
    dict with keys: strategy_cumulative, buyhold_cumulative,
    sharpe_ratio, max_drawdown, win_rate, and daily series.
    """
    daily_returns = daily_returns.values if isinstance(daily_returns, pd.Series) else daily_returns
    n = min(len(daily_returns), len(y_pred))
    daily_returns = daily_returns[:n]
    y_pred = y_pred[:n]

    strategy_daily = daily_returns * y_pred

    strategy_cumulative = (1 + strategy_daily).cumprod()
    buyhold_cumulative = (1 + daily_returns).cumprod()

    excess = strategy_daily - (risk_free / trading_days)
    sharpe = np.sqrt(trading_days) * excess.mean() / (excess.std() + 1e-9)

    running_max = np.maximum.accumulate(strategy_cumulative)
    drawdowns = (strategy_cumulative - running_max) / running_max
    max_dd = drawdowns.min()

    traded_days = y_pred == 1
    if traded_days.sum() > 0:
        wins = (daily_returns[traded_days] > 0).sum()
        win_rate = wins / traded_days.sum()
    else:
        win_rate = 0.0

    total_strategy_return = strategy_cumulative[-1] - 1
    total_buyhold_return = buyhold_cumulative[-1] - 1

    return {
        "strategy_return": round(float(total_strategy_return), 4),
        "buyhold_return": round(float(total_buyhold_return), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "strategy_cumulative": strategy_cumulative,
        "buyhold_cumulative": buyhold_cumulative,
        "strategy_daily": strategy_daily,
    }


def backtest_summary_table(backtest_result: dict) -> pd.DataFrame:
    """Format backtest results as a clean comparison table."""
    rows = [
        {"Metric": "Total Return (Strategy)", "Value": f"{backtest_result['strategy_return']:.2%}"},
        {"Metric": "Total Return (Buy & Hold)", "Value": f"{backtest_result['buyhold_return']:.2%}"},
        {"Metric": "Sharpe Ratio (Annualized)", "Value": f"{backtest_result['sharpe_ratio']:.4f}"},
        {"Metric": "Max Drawdown", "Value": f"{backtest_result['max_drawdown']:.2%}"},
        {"Metric": "Win Rate (Traded Days)", "Value": f"{backtest_result['win_rate']:.2%}"},
    ]
    return pd.DataFrame(rows)

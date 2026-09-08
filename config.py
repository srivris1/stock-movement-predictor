RANDOM_SEED = 42

TICKER = "AAPL"
DATA_START = "2019-01-01"
DATA_END = "2026-09-01"
TRAIN_TEST_SPLIT_DATE = "2025-06-01"

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
EMA_SHORT = 10
EMA_MEDIUM = 20
EMA_LONG = 50
VOLUME_SMA_PERIOD = 20
STOCHASTIC_PERIOD = 14

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
}

MIN_TRAIN_DAYS = 252
WALK_FORWARD_STEP = 63

RISK_FREE_RATE = 0.05
TRADING_DAYS_PER_YEAR = 252

FIGURES_DIR = "figures"
DATA_DIR = "data"

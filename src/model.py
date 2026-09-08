import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import config as cfg


def persistence_baseline(y_test: pd.Series) -> np.ndarray:
    """
    Persistence (naive) baseline: predict that tomorrow's direction
    equals today's direction.  y_pred[t] = y_actual[t-1].
    """
    return y_test.shift(1).fillna(y_test.iloc[0]).astype(int).values


def majority_class_baseline(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    """
    Always predict the majority class observed in the training set.
    """
    majority = y_train.value_counts().idxmax()
    return np.full(len(y_test), majority)


def get_models() -> dict:
    """Return a dict of named scikit-learn classifiers."""
    return {
        "LogisticRegression": LogisticRegression(**cfg.LOGISTIC_REGRESSION_PARAMS),
        "RandomForest": RandomForestClassifier(**cfg.RANDOM_FOREST_PARAMS),
        "GradientBoosting": GradientBoostingClassifier(**cfg.GRADIENT_BOOSTING_PARAMS),
    }


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on X_train ONLY, then transform both.
    Returns (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    return X_tr, X_te, scaler


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute directional accuracy, precision, recall, F1, and ROC-AUC.
    Returns a flat dict suitable for a comparison table row.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_set_name: str = "Raw",
) -> tuple[pd.DataFrame, dict]:
    """
    Train all models on (X_train, y_train), evaluate on (X_test, y_test).

    Also computes persistence and majority-class baselines.

    Returns:
        results_df  – comparison table (DataFrame)
        predictions – dict[model_name → y_pred array]
    """
    X_tr, X_te, scaler = scale_features(X_train, X_test)
    y_tr = y_train.values
    y_te = y_test.values

    results = []
    predictions = {}

    y_persist = persistence_baseline(y_test)
    results.append(evaluate(y_te, y_persist, "Persistence"))
    predictions["Persistence"] = y_persist

    y_majority = majority_class_baseline(y_train, y_test)
    results.append(evaluate(y_te, y_majority, "MajorityClass"))
    predictions["MajorityClass"] = y_majority

    models = get_models()
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        label = f"{name} ({feature_set_name})"
        results.append(evaluate(y_te, y_pred, label))
        predictions[label] = y_pred
        print(f"  [{label}]  Accuracy={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1']:.4f}")

    return pd.DataFrame(results), predictions


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_cls=GradientBoostingClassifier,
    model_params: dict | None = None,
    min_train: int = cfg.MIN_TRAIN_DAYS,
    step: int = cfg.WALK_FORWARD_STEP,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation.

    At each fold:
      1. Train on data[0 : split_idx]
      2. Predict data[split_idx : split_idx + step]
      3. Expand the window by `step` days

    Scaler is refitted from scratch in every fold.
    Returns a DataFrame with per-fold metrics.
    """
    if model_params is None:
        model_params = cfg.GRADIENT_BOOSTING_PARAMS

    n = len(X)
    fold_results = []
    fold_num = 0

    split_idx = min_train
    while split_idx + step <= n:
        fold_num += 1
        X_tr = X.iloc[:split_idx]
        y_tr = y.iloc[:split_idx]
        X_te = X.iloc[split_idx : split_idx + step]
        y_te = y.iloc[split_idx : split_idx + step]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = model_cls(**model_params)
        model.fit(X_tr_s, y_tr.values)
        y_pred = model.predict(X_te_s)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        fold_results.append({
            "Fold": fold_num,
            "Train_size": len(X_tr),
            "Test_size": len(X_te),
            "Accuracy": round(acc, 4),
            "F1": round(f1, 4),
        })

        split_idx += step

    df_folds = pd.DataFrame(fold_results)
    print(f"\n[walk-forward] {fold_num} folds completed")
    print(f"  Mean Accuracy: {df_folds['Accuracy'].mean():.4f}")
    print(f"  Mean F1:       {df_folds['F1'].mean():.4f}")
    return df_folds


def get_classification_report(y_true, y_pred) -> str:
    """Pretty classification report string."""
    return classification_report(y_true, y_pred, target_names=["Down", "Up"])


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix array."""
    return confusion_matrix(y_true, y_pred)

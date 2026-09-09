import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from src.data_loader import download_ohlcv, construct_target, verify_no_leakage, split_by_date
from src.features import (
    add_raw_features,
    add_technical_indicators,
    get_feature_matrix,
    RAW_FEATURE_COLS,
    ENGINEERED_FEATURE_COLS,
    ALL_FEATURE_COLS,
)
from src.model import (
    train_and_evaluate,
    walk_forward_cv,
    get_classification_report,
    get_confusion_matrix,
    scale_features,
    get_models,
)
from src.backtest import run_backtest, backtest_summary_table
from src.visualize import (
    plot_price_with_direction,
    plot_class_balance,
    plot_comparison_table,
    plot_prediction_vs_actual,
    plot_confusion_matrices,
    plot_walk_forward,
    plot_backtest,
    plot_shap_summary,
    plot_feature_importance,
)


np.random.seed(cfg.RANDOM_SEED)


def main():
    print("=" * 70)
    print("  STOCK MOVEMENT PREDICTOR")
    print(f"  Ticker: {cfg.TICKER}  |  Seed: {cfg.RANDOM_SEED}")
    print(f"  Train: {cfg.DATA_START} -> {cfg.TRAIN_TEST_SPLIT_DATE}")
    print(f"  Test:  {cfg.TRAIN_TEST_SPLIT_DATE} -> {cfg.DATA_END}")
    print("=" * 70)

    print("\n[1/10] Downloading OHLCV data ...")
    df = download_ohlcv()

    print("\n[2/10] Constructing leak-free directional target ...")
    df = construct_target(df)

    print("\n[3/10] Verifying zero forward leakage ...")
    verify_no_leakage(df)

    print("\n[4/10] Engineering features ...")
    df = add_raw_features(df)
    df = add_technical_indicators(df)

    plot_price_with_direction(df)

    print("\n[5/10] Splitting by date ...")
    train_df, test_df = split_by_date(df)

    X_train_raw, y_train = get_feature_matrix(train_df, RAW_FEATURE_COLS)
    X_test_raw, y_test = get_feature_matrix(test_df, RAW_FEATURE_COLS)
    plot_class_balance(y_train, y_test)

    print(f"\n  Train class distribution:\n{y_train.value_counts().to_string()}")
    print(f"\n  Test class distribution:\n{y_test.value_counts().to_string()}")

    print("\n[6/10] Training models on RAW features ...")
    raw_results, raw_preds = train_and_evaluate(
        X_train_raw, y_train, X_test_raw, y_test, feature_set_name="Raw"
    )

    print("\n[7/10] Training models on ENGINEERED features ...")
    X_train_eng, y_train_eng = get_feature_matrix(train_df, ALL_FEATURE_COLS)
    X_test_eng, y_test_eng = get_feature_matrix(test_df, ALL_FEATURE_COLS)
    eng_results, eng_preds = train_and_evaluate(
        X_train_eng, y_train_eng, X_test_eng, y_test_eng, feature_set_name="Engineered"
    )

    print("\n[8/10] Building four-way comparison table ...")
    four_way = pd.concat([raw_results, eng_results.iloc[2:]], ignore_index=True)
    print("\n" + four_way.to_string(index=False))
    plot_comparison_table(four_way, title="Four-Way Comparison: Baselines vs Raw vs Engineered")

    best_row = four_way.iloc[2:].sort_values("Accuracy", ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\n  Best model: {best_name}  (Accuracy={best_row['Accuracy']:.4f})")

    if best_name in eng_preds:
        best_preds = eng_preds[best_name]
        y_test_best = y_test_eng
        dates_best = X_test_eng.index
    else:
        best_preds = raw_preds[best_name]
        y_test_best = y_test
        dates_best = X_test_raw.index

    plot_prediction_vs_actual(dates_best, y_test_best.values, best_preds, model_name=best_name)

    all_preds = {}
    all_preds.update(raw_preds)
    all_preds.update(eng_preds)
    plot_confusion_matrices(y_test_best, all_preds)

    print(f"\n  Classification Report – {best_name}:")
    print(get_classification_report(y_test_best, best_preds))

    print("\n[9/10] Running walk-forward cross-validation ...")
    X_all, y_all = get_feature_matrix(df, ALL_FEATURE_COLS)
    wf_results = walk_forward_cv(X_all, y_all)
    print(wf_results.to_string(index=False))
    plot_walk_forward(wf_results)

    print("\n[10/10] Running economic backtest ...")
    test_returns = test_df["Close"].pct_change().dropna()
    common_idx = dates_best.intersection(test_returns.index)
    aligned_returns = test_returns.loc[common_idx]
    aligned_preds = pd.Series(best_preds, index=dates_best).loc[common_idx].values

    bt = run_backtest(aligned_returns, aligned_preds)
    bt_table = backtest_summary_table(bt)
    print("\n" + bt_table.to_string(index=False))
    plot_backtest(bt, model_name=best_name)

    print("\n[BONUS] Computing SHAP feature attributions ...")
    try:
        import shap
        from sklearn.ensemble import GradientBoostingClassifier
        X_tr_s, X_te_s, scaler = scale_features(X_train_eng, X_test_eng)
        gb_model = GradientBoostingClassifier(**cfg.GRADIENT_BOOSTING_PARAMS)
        gb_model.fit(X_tr_s, y_train_eng.values)

        explainer = shap.TreeExplainer(gb_model)
        shap_values = explainer.shap_values(X_te_s)

        plot_shap_summary(shap_values, ALL_FEATURE_COLS, X_te_s)
        plot_feature_importance(gb_model, ALL_FEATURE_COLS)
        print("  SHAP plots generated.")
    except Exception as e:
        print(f"  SHAP computation skipped: {e}")

    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    four_way.to_csv(os.path.join(cfg.DATA_DIR, "comparison_table.csv"), index=False)
    wf_results.to_csv(os.path.join(cfg.DATA_DIR, "walk_forward_results.csv"), index=False)
    bt_table.to_csv(os.path.join(cfg.DATA_DIR, "backtest_summary.csv"), index=False)

    plt.close("all")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Figures saved to: {os.path.abspath(cfg.FIGURES_DIR)}")
    print(f"  Data saved to:    {os.path.abspath(cfg.DATA_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

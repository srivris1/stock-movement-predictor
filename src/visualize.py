import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import config as cfg

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "up": "#2ecc71",
    "down": "#e74c3c",
    "strategy": "#3498db",
    "buyhold": "#95a5a6",
    "primary": "#2c3e50",
    "secondary": "#8e44ad",
    "accent": "#e67e22",
}


def _save(fig, filename: str):
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    path = os.path.join(cfg.FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  [fig] Saved {path}")


def plot_price_with_direction(df: pd.DataFrame, ticker: str = cfg.TICKER):
    """Plot closing price colored by actual next-day direction."""
    fig, ax = plt.subplots(figsize=(14, 5))
    up_mask = df["target"] == 1
    ax.scatter(df.index[up_mask], df["Close"][up_mask],
               c=COLORS["up"], s=4, alpha=0.6, label="Next day UP")
    ax.scatter(df.index[~up_mask], df["Close"][~up_mask],
               c=COLORS["down"], s=4, alpha=0.6, label="Next day DOWN")
    ax.set_title(f"{ticker} – Closing Price Colored by Next-Day Direction")
    ax.set_ylabel("Close ($)")
    ax.legend(markerscale=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _save(fig, "01_price_direction.png")
    return fig, ax


def plot_class_balance(y_train: pd.Series, y_test: pd.Series):
    """Side-by-side class distribution for train and test."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, title in [(axes[0], y_train, "Train"), (axes[1], y_test, "Test")]:
        counts = data.value_counts().sort_index()
        bars = ax.bar(["Down (0)", "Up (1)"], counts.values,
                       color=[COLORS["down"], COLORS["up"]], edgecolor="white", linewidth=1.5)
        ax.set_title(f"{title} Set  (n={len(data)})")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, counts.values):
            pct = v / len(data) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Class Balance Report", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "02_class_balance.png")
    return fig, axes


def plot_comparison_table(results_df: pd.DataFrame, title: str = "Four-Way Comparison"):
    """Render comparison table as a styled matplotlib table figure."""
    fig, ax = plt.subplots(figsize=(12, max(2, len(results_df) * 0.5 + 1)))
    ax.axis("off")

    col_labels = results_df.columns.tolist()
    cell_text = results_df.values.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(cell_text) + 1):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title(title, fontweight="bold", fontsize=14, pad=20)
    _save(fig, "03_comparison_table.png")
    return fig, ax


def plot_prediction_vs_actual(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
):
    """
    Plot predicted vs actual directional movement across the test window.
    Shows where the model got it right and wrong.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    correct = y_true == y_pred
    ax = axes[0]
    ax.scatter(dates[correct], y_true[correct],
               c=COLORS["up"], s=12, alpha=0.7, label="Correct", zorder=3)
    ax.scatter(dates[~correct], y_true[~correct],
               c=COLORS["down"], s=12, alpha=0.7, label="Incorrect", zorder=3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Down", "Up"])
    ax.set_title(f"Predicted vs Actual Direction – {model_name}")
    ax.legend()

    rolling_acc = pd.Series(correct.astype(float)).rolling(20, min_periods=5).mean()
    ax2 = axes[1]
    ax2.plot(dates, rolling_acc, color=COLORS["primary"], linewidth=1.5)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax2.set_ylabel("Rolling 20d Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, "04_prediction_vs_actual.png")
    return fig, axes


def plot_confusion_matrices(y_true, predictions_dict: dict):
    """Plot confusion matrices for multiple models side by side."""
    model_names = [k for k in predictions_dict if k not in ("Persistence", "MajorityClass")]
    n = len(model_names)
    if n == 0:
        return None, None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_true, predictions_dict[name])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=10)

    fig.suptitle("Confusion Matrices", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_confusion_matrices.png")
    return fig, axes


def plot_walk_forward(fold_results: pd.DataFrame):
    """Plot per-fold accuracy across walk-forward folds."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fold_results["Fold"], fold_results["Accuracy"],
           color=COLORS["primary"], alpha=0.8, label="Accuracy")
    ax.bar(fold_results["Fold"], fold_results["F1"],
           color=COLORS["accent"], alpha=0.5, width=0.4, label="F1")
    ax.axhline(fold_results["Accuracy"].mean(), color=COLORS["down"],
               linestyle="--", linewidth=1, label=f"Mean Acc={fold_results['Accuracy'].mean():.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Walk-Forward Cross-Validation – Per-Fold Performance")
    ax.legend()
    fig.tight_layout()
    _save(fig, "06_walk_forward.png")
    return fig, ax


def plot_backtest(backtest_result: dict, model_name: str = "Best Model"):
    """Plot cumulative returns: strategy vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(12, 5))
    strat = backtest_result["strategy_cumulative"]
    bh = backtest_result["buyhold_cumulative"]

    x = range(len(strat))
    ax.plot(x, strat, color=COLORS["strategy"], linewidth=2, label="Model Strategy")
    ax.plot(x, bh, color=COLORS["buyhold"], linewidth=2, label="Buy & Hold", linestyle="--")
    ax.fill_between(x, strat, 1, where=strat > 1, alpha=0.1, color=COLORS["strategy"])
    ax.axhline(1.0, color="gray", linewidth=0.5)

    ax.set_title(f"Economic Backtest – {model_name} vs Buy & Hold")
    ax.set_xlabel("Trading Days (Test Period)")
    ax.set_ylabel("Cumulative Return (1.0 = initial)")
    ax.legend()

    sr = backtest_result["strategy_return"]
    br = backtest_result["buyhold_return"]
    ax.annotate(f"Strategy: {sr:+.2%}", xy=(len(strat) - 1, strat[-1]),
                fontsize=9, fontweight="bold", color=COLORS["strategy"])
    ax.annotate(f"Buy&Hold: {br:+.2%}", xy=(len(bh) - 1, bh[-1]),
                fontsize=9, fontweight="bold", color=COLORS["buyhold"])

    fig.tight_layout()
    _save(fig, "07_backtest.png")
    return fig, ax


def plot_shap_summary(shap_values, feature_names: list, X_test_scaled):
    """Plot SHAP summary (bar + beeswarm)."""
    import shap

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plt.sca(axes[0])
    shap.summary_plot(shap_values, features=X_test_scaled,
                      feature_names=feature_names, plot_type="bar",
                      show=False, max_display=15)
    axes[0].set_title("SHAP Feature Importance (Bar)")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, features=X_test_scaled,
                      feature_names=feature_names, show=False,
                      max_display=15)
    axes[1].set_title("SHAP Value Distribution (Beeswarm)")

    fig.tight_layout()
    _save(fig, "08_shap_summary.png")
    return fig, axes


def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    """Plot scikit-learn feature importances (tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        return None, None

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color=COLORS["secondary"], alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Gini / Impurity)")
    fig.tight_layout()
    _save(fig, "09_feature_importance.png")
    return fig, ax

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/formal_25_features"
FIGURES = ROOT / "figures/formal_25_features"
FIGURES.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(
    RESULTS / "all_models_test_metrics.csv"
)

labels = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}

colors = sns.color_palette("tab10", n_colors=len(labels))
rng = np.random.default_rng(42)

# ROC曲线和Bootstrap AUC置信区间
plt.figure(figsize=(8, 7))
ci_rows = []

for color, model in zip(colors, labels):
    pred = pd.read_csv(
        RESULTS / f"{model}_test_predictions.csv"
    )

    y = pred["y_true_failure"].to_numpy()
    probability = pred["prob_failure"].to_numpy()

    fpr, tpr, _ = roc_curve(y, probability)
    auc = roc_auc_score(y, probability)

    bootstrap_auc = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        bootstrap_auc.append(
            roc_auc_score(y[idx], probability[idx])
        )

    lower, upper = np.percentile(
        bootstrap_auc, [2.5, 97.5]
    )

    ci_rows.append({
        "model": model,
        "label": labels[model],
        "test_auc": auc,
        "ci_lower": lower,
        "ci_upper": upper,
    })

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        color=color,
        label=(
            f"{labels[model]}: "
            f"{auc:.3f} ({lower:.3f}–{upper:.3f})"
        ),
    )

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Chance",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Failure Prediction")
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(
    FIGURES / "roc_curves_25_features.png",
    dpi=300,
)
plt.close()

ci = pd.DataFrame(ci_rows).sort_values(
    "test_auc", ascending=False
)
ci.to_csv(
    RESULTS / "test_auc_bootstrap_ci.csv",
    index=False,
)

# AUC及95%置信区间
plot_ci = ci.sort_values("test_auc")

plt.figure(figsize=(8, 5))
xerr = np.vstack([
    plot_ci["test_auc"] - plot_ci["ci_lower"],
    plot_ci["ci_upper"] - plot_ci["test_auc"],
])

plt.errorbar(
    plot_ci["test_auc"],
    plot_ci["label"],
    xerr=xerr,
    fmt="o",
    capsize=4,
    color="#1f77b4",
)

plt.axvline(
    0.723,
    linestyle="--",
    color="red",
    label="080 paper XGBoost AUC = 0.723",
)

plt.xlabel("Test ROC-AUC with 95% bootstrap CI")
plt.title("Model AUC Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(
    FIGURES / "auc_bootstrap_ci.png",
    dpi=300,
)
plt.close()

# 主要指标比较
plot_metrics = metrics.copy()
plot_metrics["Model"] = plot_metrics["model"].map(labels)

metric_columns = {
    "test_roc_auc": "ROC-AUC",
    "test_accuracy": "Accuracy",
    "test_precision_failure": "Precision",
    "test_recall_failure": "Recall",
    "test_f1_failure": "F1",
    "test_specificity": "Specificity",
}

long = plot_metrics.melt(
    id_vars="Model",
    value_vars=list(metric_columns),
    var_name="Metric",
    value_name="Value",
)
long["Metric"] = long["Metric"].map(metric_columns)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=long,
    x="Model",
    y="Value",
    hue="Metric",
)
plt.ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.title("Independent Test-Set Performance")
plt.tight_layout()
plt.savefig(
    FIGURES / "metrics_comparison.png",
    dpi=300,
)
plt.close()

print("=== AUC及95%置信区间 ===")
print(ci.round(4).to_string(index=False))
print("\n图片目录:", FIGURES)

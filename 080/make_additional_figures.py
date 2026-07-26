from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/formal_25_features"
OUT = ROOT / "figures/formal_25_features"
OUT.mkdir(parents=True, exist_ok=True)

models = [
    "logistic_regression",
    "random_forest",
    "svm",
    "gradient_boosting",
    "xgboost",
    "neural_network",
]

labels = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}

predictions = {
    model: pd.read_csv(
        RESULTS / f"{model}_test_predictions.csv"
    )
    for model in models
}

# 混淆矩阵
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for ax, model in zip(axes.flat, models):
    df = predictions[model]
    cm = confusion_matrix(
        df["y_true_failure"],
        df["pred_failure"],
        labels=[0, 1],
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title(labels[model])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Success", "Failure"])
    ax.set_yticklabels(["Success", "Failure"])

plt.tight_layout()
plt.savefig(
    OUT / "confusion_matrices.png",
    dpi=300,
)
plt.close()

# 校准曲线
plt.figure(figsize=(8, 7))

for model in models:
    df = predictions[model]

    observed, predicted = calibration_curve(
        df["y_true_failure"],
        df["prob_failure"],
        n_bins=8,
        strategy="quantile",
    )

    plt.plot(
        predicted,
        observed,
        marker="o",
        label=labels[model],
    )

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration",
)
plt.xlabel("Mean Predicted Failure Probability")
plt.ylabel("Observed Failure Rate")
plt.title("Calibration Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(
    OUT / "calibration_curves.png",
    dpi=300,
)
plt.close()

# 决策曲线：CV最佳模型和论文对应的XGBoost
thresholds = np.linspace(0.01, 0.80, 80)

plt.figure(figsize=(8, 6))

reference = predictions["gradient_boosting"]
y = reference["y_true_failure"].to_numpy()
n = len(y)
prevalence = y.mean()

treat_all = (
    prevalence
    - (1 - prevalence)
    * thresholds / (1 - thresholds)
)

plt.plot(
    thresholds,
    np.zeros_like(thresholds),
    linestyle="--",
    color="black",
    label="Treat none",
)
plt.plot(
    thresholds,
    treat_all,
    linestyle=":",
    color="gray",
    label="Treat all",
)

for model, color in [
    ("gradient_boosting", "#1f77b4"),
    ("xgboost", "#d62728"),
]:
    df = predictions[model]
    probability = df["prob_failure"].to_numpy()

    net_benefit = []

    for threshold in thresholds:
        pred = probability >= threshold

        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))

        benefit = (
            tp / n
            - fp / n
            * threshold / (1 - threshold)
        )
        net_benefit.append(benefit)

    plt.plot(
        thresholds,
        net_benefit,
        linewidth=2,
        color=color,
        label=labels[model],
    )

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.ylim(-0.05, 0.15)
plt.legend()
plt.tight_layout()
plt.savefig(
    OUT / "decision_curve.png",
    dpi=300,
)
plt.close()

print("附加图生成完成：")
for path in sorted(OUT.glob("*.png")):
    print(path.name, round(path.stat().st_size / 1024, 1), "KB")

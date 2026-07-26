from pathlib import Path
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/shap"
RESULTS = ROOT / "results/shap"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

split = pd.read_csv(
    ROOT / "data/splits/split_seed42.csv"
)
test_ids = split.loc[
    split["split"] == "test", "row_id"
].to_numpy()

experiments = {
    "gradient_boosting_25": {
        "data": ROOT / "data/processed/leakage_controlled_25_features.csv",
        "model": ROOT / "models/formal_25_features/gradient_boosting.joblib",
    },
    "xgboost_25": {
        "data": ROOT / "data/processed/leakage_controlled_25_features.csv",
        "model": ROOT / "models/formal_25_features/xgboost.joblib",
    },
    "xgboost_paper_like_29": {
        "data": ROOT / "data/processed/paper_like_29_features.csv",
        "model": ROOT / "models/xgb_baseline_paper_like.joblib",
    },
}

def clean_feature_name(name):
    name = re.sub(r"^(categorical|numeric)__", "", name)
    return name

for experiment, paths in experiments.items():
    print(f"\n处理: {experiment}", flush=True)

    data = pd.read_csv(paths["data"])
    X = data.drop(columns=["target_failure"])
    X_test = X.iloc[test_ids].copy()

    pipeline = joblib.load(paths["model"])
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(X_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = [
        clean_feature_name(x)
        for x in preprocessor.get_feature_names_out()
    ]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.to_csv(
        RESULTS / f"{experiment}_importance.csv",
        index=False,
    )

    plt.figure()
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title(f"SHAP Summary: {experiment}")
    plt.tight_layout()
    plt.savefig(
        OUT / f"{experiment}_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title(f"Mean Absolute SHAP: {experiment}")
    plt.tight_layout()
    plt.savefig(
        OUT / f"{experiment}_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Top 10:")
    print(
        importance.head(10)
        .round(4)
        .to_string(index=False)
    )

print("\nSHAP图全部生成完成。")

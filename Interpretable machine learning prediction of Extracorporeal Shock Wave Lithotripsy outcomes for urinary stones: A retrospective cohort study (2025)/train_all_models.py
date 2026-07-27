from pathlib import Path
from datetime import datetime
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/leakage_controlled_25_features.csv"
SPLIT = ROOT / "data/splits/split_seed42.csv"
RESULTS = ROOT / "results/formal_25_features"
MODELS = ROOT / "models/formal_25_features"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(DATA)
split = pd.read_csv(SPLIT)

target = "target_failure"
train_ids = split.loc[split["split"] == "train", "row_id"].to_numpy()
test_ids = split.loc[split["split"] == "test", "row_id"].to_numpy()

X = data.drop(columns=[target])
y = data[target].astype(int)

X_train = X.iloc[train_ids].copy()
X_test = X.iloc[test_ids].copy()
y_train = y.iloc[train_ids].copy()
y_test = y.iloc[test_ids].copy()

categorical_features = [
    "sex", "Comorbidities", "proced", "DJstent",
    "frequency", "ANALGESIA", "paintolerance",
    "antibiotic", "stonenumber", "laterality",
    "HDN", "UrineCULTURE", "location",
]

numeric_features = [
    c for c in X.columns if c not in categorical_features
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ),
        categorical_features,
    ),
    (
        "numeric",
        StandardScaler(),
        numeric_features,
    ),
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

c_values = [0.001, 0.01, 0.1, 1, 10, 100]

model_specs = {
    "logistic_regression": {
        "model": LogisticRegression(random_state=42),
        "params": [
            {
                "model__penalty": ["l1", "l2"],
                "model__C": c_values,
                "model__solver": ["liblinear", "saga"],
                "model__max_iter": [100, 200, 500],
            },
            {
                "model__penalty": ["l2"],
                "model__C": c_values,
                "model__solver": ["lbfgs", "newton-cg", "sag"],
                "model__max_iter": [100, 200, 500],
            },
            {
                "model__penalty": ["none"],
                "model__solver": ["lbfgs", "newton-cg", "sag", "saga"],
                "model__max_iter": [100, 200, 500],
            },
            {
                "model__penalty": ["elasticnet"],
                "model__C": c_values,
                "model__solver": ["saga"],
                "model__l1_ratio": [0.5],
                "model__max_iter": [100, 200, 500],
            },
        ],
    },
    "random_forest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        ),
        "params": {
            "model__n_estimators": [50, 100, 200, 500],
            "model__max_depth": [3, 5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    },
    "svm": {
        "model": SVC(
            probability=True,
            random_state=42,
        ),
        "params": {
            "model__C": c_values,
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "model__degree": [2, 3, 4],
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200, 500],
            "model__learning_rate": [0.001, 0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, 9],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.8, 0.9, 1.0],
        },
    },
    "xgboost": {
        "model": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        ),
        "params": {
            "model__n_estimators": [50, 100, 200, 500],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.001, 0.01, 0.1, 0.2],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__gamma": [0, 0.1, 0.2, 0.3],
            "model__scale_pos_weight": [1, 3, 6.92, 10],
        },
    },
    "neural_network": {
        "model": MLPClassifier(random_state=42),
        "params": {
            "model__hidden_layer_sizes": [
                (50,), (100,), (50, 50), (100, 50)
            ],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__solver": ["adam", "sgd"],
            "model__learning_rate": ["constant", "adaptive"],
            "model__max_iter": [200, 300, 500],
        },
    },
}

all_metrics = []

print("数据:", DATA)
print("训练集:", len(train_ids), "测试集:", len(test_ids))
print("训练集失败数:", int(y_train.sum()))
print("开始时间:", datetime.now().isoformat(timespec="seconds"), flush=True)

for name, spec in model_specs.items():
    print("\n" + "=" * 70, flush=True)
    print("开始模型:", name, flush=True)
    start = time.time()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", spec["model"]),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec["params"],
        n_iter=100,
        scoring="roc_auc",
        n_jobs=8,
        cv=cv,
        verbose=1,
        random_state=42,
        refit=True,
        return_train_score=True,
        error_score=np.nan,
    )

    search.fit(X_train, y_train)

    probability = search.best_estimator_.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, prediction, labels=[0, 1]
    ).ravel()

    elapsed = time.time() - start

    metrics = {
        "model": name,
        "best_cv_roc_auc": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_precision_failure": float(
            precision_score(y_test, prediction, zero_division=0)
        ),
        "test_recall_failure": float(
            recall_score(y_test, prediction, zero_division=0)
        ),
        "test_f1_failure": float(
            f1_score(y_test, prediction, zero_division=0)
        ),
        "test_specificity": float(tn / (tn + fp)),
        "test_roc_auc": float(roc_auc_score(y_test, probability)),
        "test_brier_score": float(
            brier_score_loss(y_test, probability)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "elapsed_minutes": float(elapsed / 60),
    }

    all_metrics.append(metrics)

    pd.DataFrame(search.cv_results_).to_csv(
        RESULTS / f"{name}_cv_results.csv",
        index=False,
    )

    pd.DataFrame({
        "row_id": test_ids,
        "y_true_failure": y_test.to_numpy(),
        "prob_failure": probability,
        "pred_failure": prediction,
    }).to_csv(
        RESULTS / f"{name}_test_predictions.csv",
        index=False,
    )

    with open(
        RESULTS / f"{name}_best_params.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            search.best_params_,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    joblib.dump(
        search.best_estimator_,
        MODELS / f"{name}.joblib",
    )

    pd.DataFrame(all_metrics).to_csv(
        RESULTS / "all_models_test_metrics.csv",
        index=False,
    )

    print("完成模型:", name, flush=True)
    print("最佳CV AUC:", round(search.best_score_, 4), flush=True)
    print("测试AUC:", round(metrics["test_roc_auc"], 4), flush=True)
    print("测试Recall:", round(metrics["test_recall_failure"], 4), flush=True)
    print("耗时分钟:", round(metrics["elapsed_minutes"], 2), flush=True)

print("\n全部模型完成。", flush=True)
print(
    pd.DataFrame(all_metrics)
    .sort_values("test_roc_auc", ascending=False)
    .to_string(index=False),
    flush=True,
)

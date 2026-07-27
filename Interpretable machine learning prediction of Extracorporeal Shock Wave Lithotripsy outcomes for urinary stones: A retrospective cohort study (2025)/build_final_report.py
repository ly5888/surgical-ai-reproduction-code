from pathlib import Path
import json
import platform

import numpy
import pandas
import sklearn
import xgboost
import imblearn
import shap

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/080_近似复现报告.md"

metrics = pandas.read_csv(
    ROOT / "results/formal_25_features/all_models_test_metrics.csv"
)
ci = pandas.read_csv(
    ROOT / "results/formal_25_features/test_auc_bootstrap_ci.csv"
)

labels = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}

paper_results = {
    "logistic_regression": [0.849, 0.375, 0.292, 0.328, 0.687, 0.134],
    "random_forest": [0.847, 0.391, 0.313, 0.347, 0.698, 0.131],
    "svm": [0.845, 0.368, 0.271, 0.312, 0.674, 0.139],
    "gradient_boosting": [0.853, 0.385, 0.333, 0.357, 0.705, 0.129],
    "neural_network": [0.844, 0.359, 0.292, 0.322, 0.682, 0.136],
    "xgboost": [0.851, 0.405, 0.354, 0.378, 0.723, 0.128],
}

cv_best = metrics.loc[metrics["best_cv_roc_auc"].idxmax()]
test_best = metrics.loc[metrics["test_roc_auc"].idxmax()]
xgb = metrics.loc[metrics["model"] == "xgboost"].iloc[0]

with open(
    ROOT / "results/xgb_baseline_controlled_metrics.json",
    encoding="utf-8",
) as f:
    controlled_baseline = json.load(f)

with open(
    ROOT / "results/xgb_baseline_paper_like_metrics.json",
    encoding="utf-8",
) as f:
    paper_like_baseline = json.load(f)

def top_shap(name, n=10):
    frame = pandas.read_csv(
        ROOT / f"results/shap/{name}_importance.csv"
    )
    return frame.head(n)

lines = [
    "# 080号文献近似复现报告",
    "",
    "## 一、复现对象",
    "",
    "论文：*Interpretable machine learning prediction of "
    "Extracorporeal Shock Wave Lithotripsy outcomes for urinary "
    "stones: A retrospective cohort study*（2025）。",
    "",
    "## 二、复现性质",
    "",
    "**本实验属于方法学近似复现，不是严格数值复现。**",
    "",
    "- 原论文使用1501例患者、29个特征；",
    "- 论文声明的Mendeley数据版本2目前不可访问；",
    "- 本实验使用同一作者公开的版本1数据，共1000例、47列；",
    "- 使用论文相同的六类模型、75/25分层划分、随机种子42、"
    "SMOTE、100次随机搜索、5折交叉验证和SHAP；",
    "- 因患者队列和部分变量不同，不应期待重现论文原始数值。",
    "",
    "## 三、数据检查",
    "",
    "- 原始数据：1000行 × 47列；",
    "- 治疗成功：873例；治疗失败：127例；",
    "- 失败率：12.7%；",
    "- 完全重复记录：4行，因无患者ID而予以保留；",
    "- 原始文件SHA-256："
    "`887dea2ecc1bfb4703faf5fa2156164885ff1b48cf01630143e2a58a762e42f7`；",
    "- 固定训练集：750例，其中失败95例；",
    "- 固定测试集：250例，其中失败32例。",
    "",
    "## 四、信息泄漏控制",
    "",
    "公开数据的29个原始预测字段中包含4个治疗后变量：",
    "",
    "- `number`：重复治疗次数；",
    "- `COMPLICATIONS`：并发症；",
    "- `postESWLemergency`：ESWL后急诊；",
    "- `additionalprocedure`：后续追加治疗。",
    "",
    "主分析删除上述字段，使用25个无明显泄漏特征。"
    "29特征分析仅作为敏感性分析。",
    "",
    "| XGBoost设置 | ROC-AUC | Recall | F1 |",
    "|---|---:|---:|---:|",
    f"| 无泄漏25特征 | {controlled_baseline['roc_auc']:.3f} | "
    f"{controlled_baseline['recall_failure']:.3f} | "
    f"{controlled_baseline['f1_failure']:.3f} |",
    f"| 含治疗后信息29特征 | {paper_like_baseline['roc_auc']:.3f} | "
    f"{paper_like_baseline['recall_failure']:.3f} | "
    f"{paper_like_baseline['f1_failure']:.3f} |",
    "",
    "29特征模型中，`additionalprocedure_0`的平均绝对SHAP值为1.913，"
    "排名第一，证明后续治疗信息会显著抬高预测性能。",
    "",
    "## 五、正式六模型结果（25特征）",
    "",
    "| 模型 | CV AUC | Accuracy | Precision | Recall | F1 | "
    "Specificity | Test AUC | Brier |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for _, row in metrics.sort_values(
    "test_roc_auc", ascending=False
).iterrows():
    lines.append(
        f"| {labels[row['model']]} | "
        f"{row['best_cv_roc_auc']:.3f} | "
        f"{row['test_accuracy']:.3f} | "
        f"{row['test_precision_failure']:.3f} | "
        f"{row['test_recall_failure']:.3f} | "
        f"{row['test_f1_failure']:.3f} | "
        f"{row['test_specificity']:.3f} | "
        f"{row['test_roc_auc']:.3f} | "
        f"{row['test_brier_score']:.3f} |"
    )

lines += [
    "",
    "按照训练集交叉验证AUC进行模型选择，Gradient Boosting最佳；"
    "Random Forest在独立测试集上的AUC最高，但该结果只能作为最终测试表现描述，"
    "不能在查看测试集后反向用于模型选择。",
    "",
    "## 六、AUC的Bootstrap 95%置信区间",
    "",
    "| 模型 | Test AUC | 95% CI |",
    "|---|---:|---:|",
]

for _, row in ci.sort_values("test_auc", ascending=False).iterrows():
    lines.append(
        f"| {row['label']} | {row['test_auc']:.3f} | "
        f"{row['ci_lower']:.3f}–{row['ci_upper']:.3f} |"
    )

lines += [
    "",
    "不同模型的置信区间存在明显重叠，因此不能根据当前测试集断言"
    "Random Forest显著优于其他模型。",
    "",
    "## 七、与080论文结果比较",
    "",
    "| 模型 | 论文AUC | 本次AUC | 论文Recall | 本次Recall |",
    "|---|---:|---:|---:|---:|",
]

for model in paper_results:
    paper = paper_results[model]
    reproduced = metrics.loc[metrics["model"] == model].iloc[0]
    lines.append(
        f"| {labels[model]} | {paper[4]:.3f} | "
        f"{reproduced['test_roc_auc']:.3f} | "
        f"{paper[2]:.3f} | "
        f"{reproduced['test_recall_failure']:.3f} |"
    )

lines += [
    "",
    f"原论文中XGBoost最佳，AUC为0.723；本次XGBoost AUC为"
    f"{xgb['test_roc_auc']:.3f}，但模型排名并非第一。"
    "因此，论文的数值结果和“XGBoost最佳”结论没有被严格复现。",
    "",
    "## 八、SHAP解释",
    "",
    "### Gradient Boosting前10个特征",
    "",
    "| 特征 | Mean absolute SHAP |",
    "|---|---:|",
]

for _, row in top_shap("gradient_boosting_25").iterrows():
    lines.append(
        f"| {row['feature']} | {row['mean_abs_shap']:.4f} |"
    )

lines += [
    "",
    "### XGBoost前10个特征",
    "",
    "| 特征 | Mean absolute SHAP |",
    "|---|---:|",
]

for _, row in top_shap("xgboost_25").iterrows():
    lines.append(
        f"| {row['feature']} | {row['mean_abs_shap']:.4f} |"
    )

lines += [
    "",
    "主要特征包括皮肤至结石距离（DDS）、既往泌尿外科操作、"
    "BUN、BMI、年龄、结石密度（HU）和结石大小。其方向总体具有临床合理性，"
    "但与原论文中“结石密度和大小最重要”的排序不完全一致。",
    "",
    "## 九、图表证据",
    "",
    "- `figures/formal_25_features/roc_curves_25_features.png`",
    "- `figures/formal_25_features/auc_bootstrap_ci.png`",
    "- `figures/formal_25_features/metrics_comparison.png`",
    "- `figures/formal_25_features/confusion_matrices.png`",
    "- `figures/formal_25_features/calibration_curves.png`",
    "- `figures/formal_25_features/decision_curve.png`",
    "- `figures/shap/gradient_boosting_25_summary.png`",
    "- `figures/shap/xgboost_25_summary.png`",
    "- `figures/shap/xgboost_paper_like_29_summary.png`",
    "",
    "## 十、环境",
    "",
    f"- 操作系统：{platform.platform()}",
    f"- Python：{platform.python_version()}",
    f"- NumPy：{numpy.__version__}",
    f"- pandas：{pandas.__version__}",
    f"- scikit-learn：{sklearn.__version__}",
    f"- XGBoost：{xgboost.__version__}",
    f"- imbalanced-learn：{imblearn.__version__}",
    f"- SHAP：{shap.__version__}",
    "",
    "## 十一、最终结论",
    "",
    "1. 已完成080论文的公开数据方法学近似复现；",
    "2. 六种模型、超参数搜索、独立测试、Bootstrap置信区间、"
    "SHAP、校准与决策曲线均已完成；",
    "3. 没有复现出原论文“XGBoost最佳”的模型排序；",
    "4. 结果差异主要来自数据队列不同；",
    "5. 公开数据中的治疗后变量会造成明显信息泄漏，"
    "因此25特征无泄漏分析应作为主结果；",
    "6. 若要严格复现原论文，仍需作者提供1501例版本2数据和真实分析代码。",
]

REPORT.write_text("\n".join(lines), encoding="utf-8")
print("报告已生成:", REPORT)

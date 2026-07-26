# 080 — ESWL治疗失败预测近似复现代码

## 项目简介

本目录包含080号文献的近似复现脚本。研究目标是根据患者和结石相关特征预测体外冲击波碎石术（ESWL）治疗失败。

对应论文：

> *Interpretable machine learning prediction of Extracorporeal Shock Wave Lithotripsy outcomes for urinary stones: A retrospective cohort study*（2025）

本实验使用作者公开的1000例版本1数据。原论文使用1501例版本2数据，因此本项目属于 **方法学近似复现**，不是严格数值复现。

## 文件说明

| 文件 | 用途 |
|---|---|
| `train_all_models.py` | 训练和比较六类模型，进行SMOTE、5折交叉验证和100次随机搜索 |
| `make_evaluation_figures.py` | 生成ROC曲线、Bootstrap AUC置信区间和主要指标对比图 |
| `make_additional_figures.py` | 生成混淆矩阵、校准曲线和决策曲线 |
| `make_shap_figures.py` | 生成Gradient Boosting、XGBoost及29特征敏感性模型的SHAP结果 |
| `build_final_report.py` | 汇总指标、图表、环境和复现结论，生成Markdown报告 |

## 模型

正式25特征分析比较：

1. Logistic Regression；
2. Random Forest；
3. SVM；
4. Gradient Boosting；
5. XGBoost；
6. Neural Network（MLP）。

统一设置包括：

- 75%训练集、25%测试集；
- 分层划分；
- `random_state=42`；
- SMOTE仅用于训练流程；
- 5折分层交叉验证；
- 每个模型100次随机超参数搜索；
- 固定0.5分类阈值。

## 信息泄漏控制

公开数据包含4个治疗后变量：

- `number`：重复治疗次数；
- `COMPLICATIONS`：并发症；
- `postESWLemergency`：ESWL后急诊；
- `additionalprocedure`：后续追加治疗。

正式主分析删除这些变量，使用25个无明显治疗后信息泄漏的特征。包含29个特征的结果只作为敏感性分析。

## 环境

原复现环境使用 Python 3.9。建议安装：

```bash
pip install \
  numpy pandas scipy joblib matplotlib seaborn \
  scikit-learn imbalanced-learn xgboost shap
```

## 完整项目目录要求

本目录仅包含代码。运行时需要将脚本放在完整项目的 `src/` 文件夹中：

```text
eswl080_reproduction/
├── data/
│   ├── processed/
│   │   ├── leakage_controlled_25_features.csv
│   │   └── paper_like_29_features.csv
│   └── splits/
│       └── split_seed42.csv
├── src/
│   ├── train_all_models.py
│   ├── make_evaluation_figures.py
│   ├── make_additional_figures.py
│   ├── make_shap_figures.py
│   └── build_final_report.py
├── models/
├── results/
├── figures/
└── reports/
```

脚本通过自身位置自动定位项目根目录，因此不要随意改变 `src/` 与其他目录的相对关系。

## 推荐运行顺序

在项目根目录运行：

```bash
python src/train_all_models.py
python src/make_evaluation_figures.py
python src/make_additional_figures.py
python src/make_shap_figures.py
python src/build_final_report.py
```

### 第一步：训练六类模型

```bash
python src/train_all_models.py
```

主要输入：

```text
data/processed/leakage_controlled_25_features.csv
data/splits/split_seed42.csv
```

主要输出：

```text
models/formal_25_features/
results/formal_25_features/
```

该步骤进行六类模型的超参数搜索，运行时间可能较长。

### 第二步：生成评价图

```bash
python src/make_evaluation_figures.py
```

生成ROC曲线、测试集AUC的Bootstrap 95%置信区间和指标对比图。

### 第三步：生成附加图

```bash
python src/make_additional_figures.py
```

生成：

- 混淆矩阵；
- 校准曲线；
- 决策曲线。

### 第四步：生成SHAP解释

```bash
python src/make_shap_figures.py
```

此脚本还需要以下模型已经存在：

```text
models/formal_25_features/gradient_boosting.joblib
models/formal_25_features/xgboost.joblib
models/xgb_baseline_paper_like.joblib
```

以及25特征和29特征两个处理后数据文件。

### 第五步：生成报告

```bash
python src/build_final_report.py
```

报告保存到：

```text
reports/080_近似复现报告.md
```

该脚本依赖此前生成的指标、Bootstrap置信区间、SHAP重要性文件，以及两份XGBoost基线指标JSON。如果相关文件缺失，报告脚本会报错。

## 主要输出

```text
results/formal_25_features/all_models_test_metrics.csv
results/formal_25_features/test_auc_bootstrap_ci.csv
figures/formal_25_features/roc_curves_25_features.png
figures/formal_25_features/auc_bootstrap_ci.png
figures/formal_25_features/metrics_comparison.png
figures/formal_25_features/confusion_matrices.png
figures/formal_25_features/calibration_curves.png
figures/formal_25_features/decision_curve.png
figures/shap/
reports/080_近似复现报告.md
```

## 本次复现结论

- 已完成1000例公开数据上的方法学近似复现；
- 已完成六模型比较、独立测试、Bootstrap置信区间、SHAP、校准和决策曲线；
- 正式分析使用25个无明显治疗后信息泄漏的特征；
- 没有复现原论文“XGBoost最佳”的模型排序；
- 结果差异主要来自患者队列和变量版本不同；
- 如需严格复现，仍需要作者提供1501例版本2数据和真实分析代码。


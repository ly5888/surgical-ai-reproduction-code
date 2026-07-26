# 061 — VitalDB术中低血压预测复现代码

## 项目简介

本目录保存061号文献的复现实验脚本。任务是使用 VitalDB 公开数据预测术中低血压，并比较：

- MAP阈值临床基线；
- 基于多时间尺度生命体征特征的 XGBoost 模型。

核心实验设置：

- 观察窗：10分钟；
- 提前期：2分钟；
- 预测窗：8分钟；
- 采样间隔：30秒；
- 随机种子：42；
- 数据范围：VitalDB内部验证。

原论文使用的 Grenoble/HG 外部验证数据未公开，因此本复现不包含外部验证。

## 代码来源

本目录是从作者项目的实验脚本中抽取的代码，不是一个可独立运行的完整仓库。核心的 `hp_pred` Python 包、数据下载模块和模型测试类仍需使用作者仓库：

```text
https://github.com/BobAubouin/hypotension_pred
```

论文对应分支：

```text
jbhi_XP
```

## 文件结构

```text
061/
├── dataset_build/
│   ├── 30_s_dataset.py
│   ├── base_dataset.py
│   ├── chu_dataset.py
│   └── signal_dataset.py
├── experiments/
│   ├── train_model.py
│   ├── show_results.ipynb
│   ├── show_results_vitaldb.ipynb
│   └── study_leading_time.ipynb
└── illustrations/
    └── 若干数据探索与预处理Notebook
```

### 主要入口

| 文件 | 用途 |
|---|---|
| `dataset_build/30_s_dataset.py` | 将VitalDB病例构建为30秒采样的数据集 |
| `experiments/train_model.py` | 进行100次Optuna搜索并训练XGBoost |
| `experiments/show_results_vitaldb.ipynb` | 本次VitalDB内部测试结果分析 |
| `experiments/show_results.ipynb` | 原始结果展示、PR/ROC和SHAP分析 |
| `experiments/study_leading_time.ipynb` | 提前期实验 |

## 环境准备

建议使用 Linux、Python 3.11 和独立虚拟环境：

```bash
git clone https://github.com/BobAubouin/hypotension_pred.git
cd hypotension_pred
git checkout jbhi_XP

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install jupyter optuna xgboost pandas pyarrow shap
```

将本目录的 `dataset_build/`、`experiments/` 和 `illustrations/` 放入完整仓库的 `scripts/` 下，或根据当前目录结构调整执行路径。

## 数据目录

运行代码前需要准备：

```text
data/
├── cases/                  VitalDB病例Parquet文件
├── static_data.parquet     年龄、BMI、ASA等静态变量
├── datasets/
├── models/
└── results/
```

本代码包不包含患者原始数据。

## 推荐运行顺序

所有命令都应在完整的 `hypotension_pred` 仓库根目录运行。

### 1. 下载VitalDB数据

```bash
python -m hp_pred.dataset_download
```

### 2. 构建30秒数据集

```bash
python scripts/dataset_build/30_s_dataset.py
```

预期输出：

```text
data/datasets/30_s_dataset/
├── cases/
└── meta.parquet
```

### 3. 训练XGBoost

```bash
mkdir -p data/models data/results
python scripts/experiments/train_model.py
```

模型默认保存为：

```text
data/models/xgb_30_s.json
```

如果该文件已经存在，训练脚本会直接加载模型，而不会重新进行Optuna搜索。

### 4. 分析结果

```bash
jupyter notebook scripts/experiments/show_results_vitaldb.ipynb
```

也可以运行：

```bash
jupyter notebook scripts/experiments/show_results.ipynb
```

## 输入特征

生命体征包括：

- MAP、SBP、DBP；
- 心率、呼吸频率、SpO₂、EtCO₂；
- MAC、输注晶体量和胶体量、体温。

静态变量包括：

- 年龄；
- BMI；
- ASA分级。

模型使用1分钟、3分钟和10分钟多个时间尺度上的常数项、斜率和标准差特征。

## 本次内部测试结果

| 方法 | Precision | Recall | AUPRC | AUROC |
|---|---:|---:|---:|---:|
| MAP阈值基线 | 31.92% | 28.18% | 0.2361 | 0.7195 |
| XGBoost | 47.24% | 27.54% | 0.3933 | 0.8162 |

在相近召回率下，XGBoost具有更高的Precision、AUPRC和AUROC。SHAP结果显示，模型主要参考最近1分钟、3分钟和10分钟的平均动脉压与舒张压，方向与临床常识基本一致。

## 复现边界

- 已完成VitalDB内部测试集复现；
- 未进行Grenoble/HG外部验证，因为数据未公开；
- Notebook内可能保留作者本地路径或历史输出，换机器后应重新执行相关单元格；
- 本目录未包含完整的 `hp_pred` 包，因此必须配合作者仓库使用。


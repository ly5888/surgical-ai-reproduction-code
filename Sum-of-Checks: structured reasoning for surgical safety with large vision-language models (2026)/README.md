# 017 — Sum-of-Checks 复现代码

## 项目简介

本目录包含论文 **“Sum-of-Checks: Structured Reasoning for Surgical Safety with Large Vision-Language Models”** 的独立迁移性复现代码。

复现任务是在 Endoscapes2023 胆囊切除术图像上判断 Critical View of Safety（CVS）的三个标准：

- `C1`：仅有两条管状结构进入胆囊；
- `C2`：肝胆三角内脂肪和纤维组织已清除；
- `C3`：胆囊下三分之一已与肝床分离。

本复现使用 Qwen3-VL-Plus，对比：

- `Direct`：直接判断 C1、C2、C3；
- `SoC-no-FS`：将每项标准拆分为多个检查项，再按固定权重汇总，不使用 few-shot 示例。

## 文件说明

| 文件 | 用途 |
|---|---|
| `sumofchecks_eval.py` | anchored-v1 主程序；负责分层抽样、Direct/SoC-no-FS 推理、断点续跑、指标统计和自检 |
| `sumofchecks_eval_v2.py` | non-anchored-v2；移除带有负向锚定倾向的输出示例，并支持8张独立开发集诊断 |

## 重要说明

论文作者公开仓库当时仅包含 README 和论文 PDF，没有提供完整可执行代码、原始提示词和 four-shot 推理标注。本目录中的代码属于依据论文描述完成的独立重建，不是作者原始实现。

本目录只保存程序文件。正式运行还需要：

- Endoscapes2023 图像；
- 带有 `image_path`、`c1`、`c2`、`c3` 等字段的 CSV 清单；
- 可访问 Qwen3-VL-Plus 的阿里云百炼 API；
- `manifests/` 和 `results/` 目录。

## 环境

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

## API配置

在项目根目录创建 `.env`：

```text
DASHSCOPE_API_KEY=你的API密钥
DASHSCOPE_BASE_URL=你的OpenAI兼容接口地址
DASHSCOPE_MODEL=qwen3-vl-plus
```

不要上传或公开包含真实密钥的 `.env` 文件。

## 推荐目录结构

```text
017/
├── sumofchecks_eval.py
├── sumofchecks_eval_v2.py
├── .env
├── manifests/
│   ├── test_791_seed42.csv
│   ├── test_30_stratified_seed42.csv
│   └── dev8_nonanchored_v2_seed20260720.csv
└── results/
```

CSV 中的 `image_path` 必须指向实际存在的图像文件。

## anchored-v1运行方法

先运行程序自检：

```bash
python sumofchecks_eval.py self-test
```

从791张抽样框中生成30张固定分层样本：

```bash
python sumofchecks_eval.py make-manifest
```

先用 `--dry-run` 检查路径，不调用API：

```bash
python sumofchecks_eval.py run \
  --method direct \
  --output results/stratified30_direct.jsonl \
  --dry-run
```

运行 Direct：

```bash
python sumofchecks_eval.py run \
  --method direct \
  --output results/stratified30_direct.jsonl
```

运行 SoC-no-FS：

```bash
python sumofchecks_eval.py run \
  --method soc-nofs \
  --output results/stratified30_soc_nofs.jsonl
```

相同命令可在中断后继续运行，已经成功写入 JSONL 的样本会自动跳过。

比较两种方法：

```bash
python sumofchecks_eval.py summarize \
  --direct results/stratified30_direct.jsonl \
  --soc results/stratified30_soc_nofs.jsonl \
  --output results/stratified30_comparison.json
```

## non-anchored-v2运行方法

生成与正式30张样本互斥的8张开发集：

```bash
python sumofchecks_eval_v2.py make-dev-manifest
```

分别运行 Direct-v2 和 SoC-no-FS-v2：

```bash
python sumofchecks_eval_v2.py run \
  --method direct \
  --manifest manifests/dev8_nonanchored_v2_seed20260720.csv \
  --output results/dev8_v2_direct.jsonl

python sumofchecks_eval_v2.py run \
  --method soc-nofs \
  --manifest manifests/dev8_nonanchored_v2_seed20260720.csv \
  --output results/dev8_v2_soc_nofs.jsonl
```

检查是否仍然发生“全部预测为0”的塌缩：

```bash
python sumofchecks_eval_v2.py development-report \
  --results results/dev8_v2_soc_nofs.jsonl
```

## 输出

- 推理结果采用 JSONL，每行对应一张图像；
- 输出包含真值、预测、分数、原始回答、错误信息和 token 用量；
- `summarize` 输出 Exact Accuracy，以及 C1–C3 的 Accuracy、Precision、Recall、Specificity、F1和混淆矩阵计数；
- v2运行会额外生成 `.config.json`，记录提示版本、提示哈希和运行配置。

## 本次复现结论

技术管线可以重复运行，但在30张分层样本和8张独立开发样本中，没有观察到可靠的CVS判别能力，也没有严格复现论文报告的性能提升。该结果应表述为 **Sum-of-Checks在Qwen3-VL-Plus上的独立迁移性重建**，而非原论文的严格数值复现。


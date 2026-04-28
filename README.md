# 						大模型后训练算法对比实验

基于 **Qwen2.5-1.5B-Instruct** 的强化学习后训练对比实验框架，在 MATH 数据集上对比 **GRPO**、**GSPO**、**DAPO** 三种 RL 算法的训练效果。

---

## 项目简介

本项目旨在复现并对比三种主流的 LLM 强化学习后训练算法：

| 算法           | 全称                                 | 特点                                                        |
| -------------- | ------------------------------------ | ----------------------------------------------------------- |
| **GRPO** | Group Relative Policy Optimization   | 组内相对奖励归一化，去除critic model     |
| **GSPO** | Group Sequence Policy Optimization   | 序列级别的重要性采样 |
| **DAPO** | Dynamic Sampling Policy Optimization | 动态采样过滤全对/全错组，支持 overlong penalty，clip上限调高 |

核心特性：

- **统一训练框架**：三种算法共享数据加载、rollout 生成、checkpoint 管理、日志监控等基础设施
- **LoRA 微调**：默认使用 LoRA 训练，支持自动 merge 为完整模型
- **vLLM 加速**：rollout 生成与评测推理均支持 vLLM 高性能引擎
- **规则式数学 Reward**：基于 SymPy 的符号等价判断 + 格式奖惩
- **训练中验证**：支持定期在验证集上评测，自动保存最优 checkpoint 并支持 early stopping
- **WandB 监控**：完整的训练指标实时追踪

---

## 项目结构

```
post_training_contrast/
├── configs/                        # 配置文件
│   ├── train/                      # 训练配置
│   │   ├── grpo_math.json
│   │   ├── gspo_math.json
│   │   └── dapo_math.json
│   ├── eval/                       # 评测配置（仅作参考，推荐 CLI）
│   │   ├── baseline_math500.json
│   │   ├── grpo_math.json
│   │   ├── gspo_math.json
│   │   └── dapo_math.json
│   └── deepspeed/                  # DeepSpeed 配置
│       ├── zero3.json
│       └── zero3_offload.json
├── scripts/                        # 入口脚本
│   ├── train.py                    # 训练入口
│   ├── evaluate.py                 # 评测入口
│   ├── post_train_eval.py          # 训练后自动评测（含 LoRA merge）
│   ├── analyze_eval_badcases.py    # 错误样本分析
│   ├── compare_eval_results.py     # 多算法对比
│   ├── plot.py                     # 可视化绘图
│   ├── preprocess_datasets.py      # 原始数据预处理
│   ├── build_math_dev_split.py     # 构建 train/dev 划分
│   ├── build_math_stratified_splits.py  # 分层采样划分
│   └── test_vllm_rollout.py        # vLLM rollout 单测
├── src/post_training_contrast/     # 核心源码
│   ├── algorithms/                 # RL 算法实现
│   │   ├── base.py                 # 算法基类（advantage 计算、loss 公式）
│   │   ├── grpo.py
│   │   ├── gspo.py
│   │   └── dapo.py
│   ├── data/                       # 数据加载与划分
│   │   └── math_dev_split.py       # MATH 数据集分层划分工具
│   ├── config.py                   # 训练/评测配置 dataclass
│   ├── trainer.py                  # GRPO/GSPO 主训练循环
│   ├── dapo_trainer.py             # DAPO 专用训练器
│   ├── training_runtime.py         # 训练运行时公共工具
│   ├── evaluator.py                # 评测器
│   ├── rollout.py                  # vLLM rollout 生成
│   ├── policy.py                   # 策略模型封装
│   ├── reward.py                   # 数学 reward（规则式）
│   ├── math_utils.py               # 数学答案解析与符号等价
│   └── visualization.py            # 可视化工具函数
└── requirements.txt                # Python 依赖
```

---

## 环境要求

| 项目   | 要求                                    |
| ------ | --------------------------------------- |
| Python | >= 3.10                                 |
| CUDA   | >= 12.1（推荐），11.8 也可              |
| GPU    | NVIDIA A800 / A100（建议 >= 80GB 显存） |
| torch  | 2.4.0                                   |
| vLLM   | 0.5.4                                   |

---

## 安装依赖

```bash
# 1. 克隆仓库
git clone https://github.com/Linxinrong8023/grpo-gspo-dapo_reproduce.git
cd grpo-gspo-dapo_reproduce

# 2. 创建并激活 conda 环境
conda create -n rl_math python=3.10 -y
conda activate rl_math

# 3. 安装依赖
pip install -r requirements.txt
```

> **注意**：`torch` 版本必须与服务器 CUDA 版本匹配。`requirements.txt` 中默认适配 CUDA 12.1+。如需 CUDA 11.8，请参照文件注释调整 torch 版本。

> **FlashAttention-2（可选）**：安装 `flash-attn` 包可加速训练中的 attention 计算。训练配置默认 `attn_implementation: "auto"`，有则用，无则自动回退到 SDPA。

---

## 数据准备

### Step 1：下载原始数据

克隆仓库后 `datasets/raw/` 目录结构已存在，只需下载数据文件到对应位置：

```bash
# MATH 训练集（7 个主题，~7500 条）
huggingface-cli download --repo-type dataset lighteval/MATH --local-dir datasets/raw/math

# MATH500 测试集（500 条）
huggingface-cli download --repo-type dataset HuggingFaceH4/MATH-500 --local-dir datasets/raw/math500

# GSM8K 测试集
huggingface-cli download --repo-type dataset openai/gsm8k --local-dir datasets/raw/gsm8k
```

> 国内环境如果 HuggingFace 下载慢，可以使用 [hf-mirror.com](https://hf-mirror.com) 镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### Step 2：预处理为标准 JSONL

```bash
python scripts/preprocess_datasets.py
```

输出到 `datasets/processed/`：

- `math_full/full.jsonl` — 完整 MATH 训练集（~7500 条）
- `math500_test/test.jsonl` — MATH500 测试集（500 条）
- `gsm8k_test/test.jsonl` — GSM8K 测试集

### Step 3：构建 train/val 划分

从完整的 7500 条 MATH 数据中，按 type × level 分层采样切出 500 条作为验证集，剩余约 7000 条作为训练集：

```bash
python scripts/build_math_stratified_splits.py
```

输出：

- `datasets/processed/math_train/train_pool_7000_after_val.jsonl` — 训练集（~7000 条）
- `datasets/processed/math_dev/dev_500_stratified.jsonl` — 分层验证集（500 条）

### Step 4：下载模型

将预训练模型放到 `models/` 目录下：

```bash
#可以从 HuggingFace、modelscope 下载
#示例：从 HuggingFace 下载
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
```

---

## 配置说明

项目采用 **JSON 配置文件 + CLI 参数覆盖** 的机制。CLI 参数优先级高于配置文件，配置文件优先级高于代码默认值。

### 训练配置

训练配置文件位于 `configs/train/`，以 `grpo_math.json` 为例说明关键参数：

```jsonc
{
  // ── 数据与模型 ──
  "model": "models/Qwen2.5-1.5B-Instruct",       // 基座模型路径
  "dataset_path": "datasets/processed/math_train/train_pool_7000_after_val.jsonl",
  "output_dir": "outputs/train/grpo_math_7000",   // 训练输出目录
  "algorithm_name": "grpo",                        // 算法：grpo / gspo / dapo

  // ── 训练超参 ──
  "learning_rate": 1e-05,
  "lr_schedule": "cosine",                         // 学习率调度：cosine / linear / constant
  "lr_warmup_steps": 10,
  "num_epochs": 2,
  "prompt_batch_size": 32,                         // 每步采样的 prompt 数
  "group_size": 8,                                 // 每个 prompt 生成的 response 数
  "mini_batch_size": 8,                            // 策略更新的 mini-batch 大小
  "gradient_accumulation_steps": 2,
  "max_grad_norm": 1.0,
  "max_new_tokens": 1280,                          // rollout 最大生成 token 数
  "temperature": 1.0,

  // ── RL 参数 ──
  "clip_epsilon": 0.2,                             // PPO clip 范围
  "kl_beta": 0.01,                                 // KL 惩罚系数

  // ── LoRA ──
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,

  // ── Checkpoint ──
  "checkpoint_interval": 50,                       // 每 N 步保存一次
  "save_best_checkpoint": true,
  "best_checkpoint_metric": "val_accuracy",
  "early_stopping_patience": 8,

  // ── 训练中验证 ──
  "eval_dataset_path": "datasets/processed/math_dev/dev_500_stratified.jsonl",
  "eval_interval": 10,                             // 每 N 步评测一次验证集
  "val_backend": "vllm",

  // ── WandB ──
  "wandb_enabled": true,
  "wandb_project": "post_training_contrast",
  "wandb_run_name": "grpo_qwen2.5_1.5b_instruct",

  // ── vLLM 显存 ──
  "gpu_memory_utilization": 0.3                    // vLLM rollout 使用的 GPU 显存比例
}
```

**DAPO 专属参数**（仅在 `algorithm_name: "dapo"` 时生效）：

| 参数                                 | 默认值   | 说明                                  |
| ------------------------------------ | -------- | ------------------------------------- |
| `dapo_use_dynamic_sampling`        | `true` | 启用动态采样，过滤全对/全错 prompt 组 |
| `dapo_use_overlong_penalty`        | `true` | 超长 response 惩罚                    |
| `dapo_clip_epsilon_low`            | `0.2`  | 不对称 clip 下界                      |
| `dapo_clip_epsilon_high`           | `0.28` | 不对称 clip 上界                      |
| `dapo_gen_prompt_batch_size`       | `32`   | 动态采样时每批候选 prompt 数          |
| `dapo_max_num_gen_batches`         | `6`    | 动态采样最大尝试批次                  |
| `dapo_min_mixed_prompt_batch_size` | `24`   | 动态采样的最少有效 prompt 数          |

### 评测配置

评测配置文件位于 `configs/eval/`，示例：

```jsonc
{
  "dataset_path": "datasets/processed/math500_test/test.jsonl",
  "dataset_name": "math500",
  "model": "outputs/train/grpo_math/checkpoint_best",  // 模型或 checkpoint 路径
  "output_path": "outputs/eval/grpo_math/math500.json",
  "max_new_tokens": 2048,
  "batch_size": 4,
  "device": "auto",
  "dtype": "auto"
}
```

### DeepSpeed 配置（本项目未使用，全量微调可使用）

位于 `configs/deepspeed/`，提供两种 ZeRO-3 配置：

- `zero3.json` — 标准 ZeRO-3（推荐）
- `zero3_offload.json` — ZeRO-3 + CPU offload（显存不足时使用）

---

## 使用指南

### 1. 训练

```bash
# 使用 GRPO 算法训练
python scripts/train.py --config configs/train/grpo_math.json

# 使用 GSPO 算法训练
python scripts/train.py --config configs/train/gspo_math.json

# 使用 DAPO 算法训练
python scripts/train.py --config configs/train/dapo_math.json
```

**CLI 参数覆盖配置文件**（常用于调试）：

```bash
# 限制最大训练步数
python scripts/train.py --config configs/train/grpo_math.json --max-steps 100

# 覆盖学习率和输出目录
python scripts/train.py --config configs/train/grpo_math.json \
    --learning-rate 5e-6 \
    --output-dir outputs/train/grpo_lr5e6

# 禁用时间戳子目录（直接写入 output_dir）
python scripts/train.py --config configs/train/grpo_math.json --no-timestamp-output-dir
```

训练输出结构：

```
outputs/train/grpo_math_7000/<timestamp>/
├── train.log                # 训练日志
├── train_summary.json       # 训练摘要（最终指标 + 配置快照）
├── checkpoint_best/         # 验证集最优 checkpoint
├── checkpoint_final/        # 最终 checkpoint
├── checkpoint_step50/       # 定期保存的 checkpoint
├── checkpoint_step100/
└── diagnostics/             # 诊断样本
```

### 2. 评测

推荐使用 CLI 参数 + vLLM 加速进行评测。项目提供两个评测脚本：

- `scripts/evaluate.py` — 通用评测，手动指定模型路径，适合 **Baseline 评测**
- `scripts/post_train_eval.py` — 训练后评测，自动读取训练输出、merge LoRA、选择 checkpoint，适合 **训练后模型评测**

#### Baseline 评测（未训练的原始模型）

```bash
python scripts/evaluate.py \
    --model models/Qwen2.5-1.5B-Instruct \
    --dataset-path datasets/processed/math500_test/test.jsonl \
    --dataset-name math500 \
    --output-path outputs/eval/baseline/math500.json \
    --max-new-tokens 2048 \
    --batch-size 128 \
    --use-vllm \
    --gpu-memory-utilization 0.9
```

#### 训练后模型评测（以 GRPO 为例）

`post_train_eval.py` 会自动从 `train_summary.json` 获取 base model 路径，如果 checkpoint 是 LoRA adapter 则自动 merge 为完整模型后评测：

```bash
# 评测 best checkpoint
python scripts/post_train_eval.py \
    --train-output-dir outputs/train/grpo_math_7000/<timestamp> \
    --checkpoint best \
    --batch-size 128 \
    --use-vllm \
    --gpu-memory-utilization 0.9

# 评测 final checkpoint
python scripts/post_train_eval.py \
    --train-output-dir outputs/train/grpo_math_7000/<timestamp> \
    --checkpoint final \
    --batch-size 128 \
    --use-vllm \
    --gpu-memory-utilization 0.9

# 评测指定步数的 checkpoint
python scripts/post_train_eval.py \
    --train-output-dir outputs/train/grpo_math_7000/<timestamp> \
    --checkpoint step100 \
    --use-vllm \
    --gpu-memory-utilization 0.9

# 指定 LoRA merge 输出目录（避免占满系统盘）
python scripts/post_train_eval.py \
    --train-output-dir outputs/train/grpo_math_7000/<timestamp> \
    --merged-dir /data/merged_models/grpo_best \
    --use-vllm \
    --gpu-memory-utilization 0.9

# 如果训练被中断、只留下 checkpoint_best 且没有 train_summary.json
python scripts/post_train_eval.py \
    --train-output-dir outputs/train/grpo_math_7000/<timestamp> \
    --checkpoint best \
    --base-model models/Qwen2.5-1.5B-Instruct \
    --use-vllm \
    --gpu-memory-utilization 0.9
```

> **说明**：`configs/eval/` 下的 JSON 配置文件仅作参考，字段不完整，正式评测请使用上述 CLI 命令。

### 3. BadCase 分析

分析评测结果中的错误样本，自动分类错误原因：

```bash
python scripts/analyze_eval_badcases.py \
    --input outputs/eval/gspo/math500.json \
    --name gspo
```

**错误分类类别**：

| 类别                                           | 说明                                   |
| ---------------------------------------------- | -------------------------------------- |
| `format_or_parse_fail`                       | 格式错误，无法从 response 中解析出答案 |
| `truncated_with_parsed_answer`               | 生成被截断，但仍解析出了答案           |
| `likely_false_negative_degree_unit`          | 度数单位差异导致的假阴性               |
| `likely_false_negative_equation_rhs`         | 方程右侧提取差异导致的假阴性           |
| `likely_false_negative_latex_frac_shorthand` | LaTeX 分数简写差异导致的假阴性         |
| `wrong_answer_parsed`                        | 答案解析成功但确实答错                 |

输出目录：

```
outputs/eval/badcases/gspo/
├── summary.json                                  # 错误分布统计
├── format_or_parse_fail.jsonl                     # 各类错误样本明细
├── wrong_answer_parsed.jsonl
└── likely_false_negative_*.jsonl
```

### 4. 多算法对比

逐题对比两个算法的评测结果：

```bash
# 默认对比 GRPO vs GSPO
python scripts/compare_eval_results.py

# 自定义对比
python scripts/compare_eval_results.py \
    --left outputs/eval/grpo/math500.json \
    --right outputs/eval/dapo/math500.json \
    --left-name grpo \
    --right-name dapo \
    --output-dir outputs/eval/comparisons/grpo_vs_dapo
```

输出：

```
outputs/eval/comparisons/grpo_vs_gspo_math500/
├── summary.json          # 对比统计（含 by-level / by-subject 切片）
├── summary.md            # Markdown 格式的对比报告
├── both_correct.jsonl     # 两个算法都对的题目
├── both_wrong.jsonl       # 两个算法都错的题目
├── left_only_correct.jsonl   # 仅左侧算法做对
└── right_only_correct.jsonl  # 仅右侧算法做对
```

### 5. 可视化绘图

```bash
# 画训练曲线（可以直接在wandb下载）
python scripts/plot.py training \
    --summary outputs/train/grpo_math_7000/<timestamp>/train_summary.json \
    --save outputs/figures/grpo_training_curve.png

# 画评测结果分布
python scripts/plot.py eval \
    --result outputs/eval/grpo/math500.json \
    --save outputs/figures/grpo_eval_summary.png

# 多算法准确率对比
python scripts/plot.py compare \
    --grpo outputs/eval/grpo/math500.json \
    --gspo outputs/eval/gspo/math500.json \
    --dapo outputs/eval/dapo/math500.json \
    --save outputs/figures/algorithm_comparison.png
```

> 加 `--no-show` 可跳过 matplotlib 弹窗，仅保存图片文件（适合远程服务器）。

### 6. vLLM Rollout 测试

独立测试 vLLM rollout 引擎：

```bash
# 单条 prompt 测试
python scripts/test_vllm_rollout.py \
    --model models/Qwen2.5-1.5B-Instruct \
    --prompt "Solve: What is 2+2?" \
    --max-new-tokens 256

# 从文件批量测试
python scripts/test_vllm_rollout.py \
    --model models/Qwen2.5-1.5B-Instruct \
    --prompt-file prompts.txt \
    --num-samples 4 \
    --temperature 0.7
```

---

## 核心模块说明

| 模块               | 文件                   | 职责                                                                      |
| ------------------ | ---------------------- | ------------------------------------------------------------------------- |
| **配置**     | `config.py`          | `TrainConfig` 和 `EvalConfig` 的 dataclass 定义，JSON + CLI 解析      |
| **训练器**   | `trainer.py`         | GRPO / GSPO 主训练循环：rollout → reward → advantage → policy update  |
| **DAPO 训练器** | `dapo_trainer.py` | DAPO 专用训练器，支持动态采样与 overlong penalty                          |
| **训练运行时** | `training_runtime.py` | 训练公共工具（checkpoint 管理、日志、WandB 等）                           |
| **算法**     | `algorithms/`        | GRPO / GSPO / DAPO 的 advantage 计算和 loss 函数实现                      |
| **Rollout**  | `rollout.py`         | vLLM 推理引擎封装，支持 LoRA 动态加载、显存管理                           |
| **策略**     | `policy.py`          | HuggingFace 模型加载，LoRA 注入，前向传播                                 |
| **Reward**   | `reward.py`          | 数学答案判分：正确/错误 reward + 格式惩罚                                 |
| **数学工具** | `math_utils.py`      | LaTeX 解析、SymPy 符号等价判断、答案提取                                  |
| **评测器**   | `evaluator.py`       | 批量生成 + 打分 + 结果汇总，支持 HF 和 vLLM 两种后端                      |
| **可视化**   | `visualization.py`   | 训练曲线、评测结果、算法对比的绘图函数                                    |

---

## WandB 监控

训练配置中设置 `"wandb_enabled": true` 即可启用：

```jsonc
{
  "wandb_enabled": true,
  "wandb_project": "post_training_contrast",
  "wandb_run_name": "grpo_qwen2.5_1.5b_instruct"
}
```

首次使用需要登录：

```bash
wandb login
```

追踪的关键指标包括：

- `train/loss` — 策略损失
- `train/kl_divergence` — 新旧策略 KL 散度
- `train/mean_reward` — 平均奖励
- `val/accuracy` — 验证集准确率
- `train/format_fail_rate` — 格式错误率
- `train/truncation_rate` — 截断率

---

## 常见问题

<details>
<summary><b>Q: 训练时显存不足 (OOM)</b></summary>

1. 降低 `gpu_memory_utilization`（vLLM rollout 占用比例）
2. 减小 `prompt_batch_size` 或 `group_size`
3. 增大 `gradient_accumulation_steps` 来代替大 batch
4. 使用 `configs/deepspeed/zero3_offload.json` 开启 CPU offload
5. 确保 `use_lora: true` 以大幅减少可训练参数量

</details>

<details>
<summary><b>Q: vLLM 与 torch 版本不兼容</b></summary>

`vllm==0.5.4` 严格依赖 `torch==2.4.0`。请勿随意升级 torch 版本。如遇冲突，建议在干净的虚拟环境中重新安装。

</details>

<details>
<summary><b>Q: 如何在没有 GPU 的环境评测？</b></summary>

设置 `"device": "cpu"`（仅适合极小的测试，正式评测需要 GPU）。

</details>

<details>
<summary><b>Q: 训练中断后如何恢复？</b></summary>

目前不支持从 checkpoint 恢复训练。如需此功能，请使用最近的 checkpoint 作为新的 base model 重新开始训练。

</details>

---

## License

本项目仅用于个人学习目的。

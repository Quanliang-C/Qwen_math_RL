# GSM8K 数学推理强化学习系统（Qwen2.5-Math-1.5B）

> *课程背景：本项目以 Stanford CS336 Assignment 5 为起点，仅保留了最初的测试脚本与流程提示。强化学习训练管线、策略优化逻辑、监控与部署均由本人自研实现。*

本仓库记录了我在 **GSM8K** 数学推理任务上构建的完整强化学习验证系统。核心目标是探索小参数推理模型 **Qwen2.5-Math-1.5B** 是否能够在不依赖闭源 Reward Model 或大模型蒸馏的情况下，仅凭强化学习获得高质量的 Chain-of-Thought 与高精度答案。系统在严格 0-shot 评测设定下，将 base model 的 pass@1 accuracy（5.4%）提升至 82%~87%，显著超越多个更大参数的公开基线模型。

---

## 1. 系统概览

整个强化学习流程由 [`cs336_alignment/grpo_with_checkpoint.py`](./cs336_alignment/grpo_with_checkpoint.py) 统一调度，覆盖以下关键组件：

- **离线模型拉取**：使用 `huggingface_hub.snapshot_download` 以离线模式加载 Qwen2.5-Math-1.5B 权重，确保在无网络环境下可复现。
- **高效采样**：依托 `vLLM` 自定义初始化（包含 FlashAttention 配置、FlashInfer 关闭、手动设定 worker multiprocessing 策略），以组为单位批量生成推理轨迹。
- **自动奖励**：通过 `drgrpo_grader.r1_zero_reward_fn` 对模型输出进行解析校验，返回 0/1 正确性信号，实现纯数学任务的 verified reward。
- **策略优化**：实现 REINFORCE+baseline 与 Dr GRPO（ratio clipping）两类策略梯度算法，自研优势函数归一化逻辑，兼顾稳定性与样本效率。
- **断点恢复**：支持完整的 checkpoint 保存与恢复（模型、优化器、LR scheduler、RNG 状态、wandb 断点），便于跨 GPU 资源迁移与长时间训练。
- **训练监控**：利用 Weights & Biases (`wandb`) 对优势函数、奖励、token 长度、token entropy、优化器学习率等指标进行全流程可视化追踪。

![Token entropy trend placeholder](./docs/figures/token_entropy_placeholder.png)

> *图 1：预留的 token entropy 随训练步数变化图示位。*

![Evaluation accuracy placeholder](./docs/figures/eval_accuracy_placeholder.png)

> *图 2：预留的评测准确率随训练步数变化图示位。*

---

## 2. 强化学习训练流程

### 2.1 数据与 Rollout 流程
- 通过 `get_gsm8k_train_ready_prompts()` 构建 prompt 与纯净答案对，形成严格的 0-shot 训练集。
- 使用 `SamplingParams` 设定温度、最小/最大生成 token 数、stop token、随机种子等，保证组内一致性。
- 每个 rollout batch 包含 `group_size = 8` 条轨迹，按 prompt 分组生成，确保对同一题目可比较多个解答。

### 2.2 优势函数与奖励计算
- `compute_group_normalized_rewards` 返回组内 baseline 的优势值，支持是否启用标准化（本实验默认关闭 `use_std_normalization=False`）。
- raw reward 直接来自 verified reward，避免额外的 reward shaping。
- 通过 wandb 记录 `raw_reward/mean`、`advantage/mean`、`advantage/max` 等统计量，实时观察样本难度与优势分布。

### 2.3 策略更新与梯度控制
- 使用 `bitsandbytes` 的 `PagedAdamW8bit` 优化器，结合 `bfloat16` 精度以降低显存占用。
- 训练 batch 按 `gradient_accumulation_steps = 32` 进行梯度累积，实现 `train_batch_size = 64` 的等效大批量训练。
- `epoch_per_rollout_batch = 3`：每轮 rollout 的轨迹重复利用三次，第二轮开始使用 GRPO ratio clipping（`grpo_clip_range = 0.30`）。
- 若某一 micro batch 优势绝对值低于阈值则跳过更新，避免噪声梯度对训练稳定性造成影响。
- `get_response_log_probs` 返回 token 级 log prob 与 entropy，`get_response_log_probs_tensor_and_response_mask` 构造有效 token mask 以对齐不同序列长度，保证 loss 计算的稳定性。

### 2.4 调度与评估
- 每完成一次 `train_step`，都会记录 `token/*` 统计信息；每隔 3 步触发一次 `grpo_evaluate.evaluate` 进行 0-shot GSM8K 全量评估。
- 每 30 步保存一次 checkpoint（模型参数 + 训练状态 + wandb 记录），支持在 `start_train_step` 指定断点恢复，适应跨机器训练。

---

## 3. 资源效率与可扩展性

- **单卡验证**：核心算法在单卡 NVIDIA L4 (24GB) 上完成，使用显存分段管理、rollout 与更新阶段解耦策略，实现 >200 GPU 小时的稳定实验。
- **高算力迁移**：同一套代码在 NVIDIA H200 上可直接扩展至单轮约 6.4k 轨迹（约 50 GPU 小时），保持相同优化步数的收敛速度与最终精度。
- **算力适配**：`gpu_memory_utilization = 0.80`、`sampling_max_tokens = 512`、`rollout_batch_size = 64` 等超参可快速调节，适配不同显存条件。

---

## 4. 主要实验结果

| 训练策略 | pass@1 Accuracy (严格 0-shot) | 训练步骤 | 备注 |
| --- | --- | --- | --- |
| Base (未训练) | 5.4% | - | Qwen2.5-Math-1.5B 原始表现 |
| REINFORCE + baseline | **≈87%** | ~200 | 无 KL/entropy 正则，仅依靠优势函数设计 |
| Dr GRPO (clip=0.30) | ≈82% | ~200 | 二阶段 ratio clipping，稳定性更优 |

![Baseline comparison placeholder](./docs/figures/accuracy_baseline_placeholder.png)

> *图 3：预留的不同超参数配置下的准确率对比图。*

---

## 5. 断点恢复与可重复性

`grpo_with_checkpoint.py` 对训练状态的保存恢复进行了完整封装：

1. `save_checkpoint()` 将模型权重、优化器、LR scheduler、随机数状态、优化步计数统一序列化。
2. 设置 `Load_From_Checkpoint=True` 且指定 `start_train_step` 和 `checkpoint_dir` 后，自动恢复训练环境。
3. 为确保跨环境一致性，还会恢复 `random`、`numpy`、`torch` 与 `torch.cuda` 的 RNG 状态，并关闭 CUDNN 的非确定性加速。

该设计支持在不同 GPU 或多次实验之间无缝切换，同时保证评估结果的可重复性。

---

## 6. 可视化与监控

- **W&B Metrics**：`train_step`、`optimizer_step` 作为全局自定义 step 指标，`raw_reward/*`、`advantage/*`、`token/*`、`eval_metrics/*`、`total_loss` 等关键指标都会随训练自动上传。
- **Token Entropy 分析**：通过 `get_response_log_probs(..., return_token_entropy=True)` 获取 token-level entropy，监控推理链的探索/收敛状态。
- **评估日志**：每次评估均记录准确率与正确题目数，支持将 RL 收敛曲线与 SFT、Expert Iteration 等方法进行对比。

---

## 7. 复现实验步骤

1. **环境准备**：
   ```bash
   uv sync --no-install-package flash-attn
   uv sync
   ```
2. **准备模型权重**：提前将 `Qwen/Qwen2.5-Math-1.5B` 下载至本地缓存目录，确保 `HF_HUB_OFFLINE=1` 时可直接加载。
3. **启动训练**：
   ```bash
   uv run python cs336_alignment/grpo_with_checkpoint.py
   ```
4. **监控训练**：在 wandb 仪表盘中观察 token entropy、raw reward、eval accuracy 等曲线，并将上文预留图表替换为真实实验截图。
5. **评估结果**：脚本会在 `outputs/` 目录写入评估日志，并在终端输出准确率，便于与基线对比。

---

## 8. 后续工作展望

- **Reward 设计**：探索更细粒度的部分分奖励与多维度奖励信号（如步骤正确性、格式合规性）。
- **自适应超参**：结合 token entropy 与 eval metric 曲线，实现自动调整学习率、clip range 等策略。
- **分布外泛化**：进一步验证在 AIME24、MATH500 等更具挑战的数据集上的迁移能力。
- **基础设施**：完善多 GPU 并行 rollout、引入梯度检查点、提升日志可视化的自动化程度。

---

## 9. 文件结构

```
.
├── cs336_alignment/
│   └── grpo_with_checkpoint.py   # 强化学习主脚本（rollout、训练、评估、断点恢复）
├── outputs/                      # 训练与评估日志输出目录（运行后生成）
├── docs/figures/                 # 预留的可视化图表存放目录
├── scripts/                      # 辅助脚本（如数据清洗、评估）
└── ...
```

- 核心逻辑集中在 `grpo_with_checkpoint.py`，其中引入的工具函数位于同目录下的 utility 模块。
- `tests/` 与课程原始仓库提供的基础测试脚本可用于快速 sanity check。

---

欢迎通过 Issue 或 PR 与我交流更多的强化学习推理实践经验。

English | [简体中文](README_CN.md)

# GSM8K Mathematical Reasoning Reinforcement Learning System (Qwen2.5‑Math‑1.5B)

This repository implements a reinforcement learning (RL) training and evaluation system for mathematical reasoning. The objective is to rigorously test whether the small reasoning model **Qwen2.5‑Math‑1.5B** can, using only an automatically verifiable reward (verified reward), substantially improve both final-answer accuracy and chain-of-thought quality, without relying on a proprietary reward model or teacher distillation from a larger model.

Under a strict zero-shot evaluation protocol (both output format and final numeric value must be correct), the system improves the pass@1 accuracy of the same base model from roughly **5.4%** to about **83%–87%**, surpassing larger open-source reasoning models (e.g., Llama‑3.1‑8B‑Instruct ~41%, Qwen3‑8B in “thinking” mode ~15%).

> Note: This project was initially inspired by Stanford CS336 Assignment 5 and reuses only basic testing scripts and some bootstrapping. The core RL pipeline, policy optimization logic, advantage design, memory adaptation, checkpointing, monitoring, and reproducibility are independently implemented.

---

## 0. Overview

The system is orchestrated by `cs336_alignment/grpo_with_checkpoint.py` and consists of the following components:

- Prompting constraints for reasoning (format-as-policy)
- Batched rollout generation (vLLM)
- Automated reward computation (verified reward)
- Policy gradient optimization (REINFORCE+baseline, Dr GRPO)
- Training stability and memory adaptation
- Checkpointing and reproducible experiments
- Rigorous evaluation and baseline comparisons

The following sections elaborate each component in order.

---

## 1. Prompting Specification

During both training and evaluation, the model must produce answers in a unified format. The dialogue-style prompt template is:

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer>\boxed{}</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Only pure answer should be placed inside <answer></answer>. For example, <answer>Tim has 3 apples</answer> is wrong, <answer>\boxed{3}</answer> is correct for the case where ground truth is 3.

User: {question}
Assistant:
```

Requirements:

- The chain-of-thought must appear inside `<think> ... </think>`.
- The final answer must appear inside `<answer> ... </answer>` and be a mathematical quantity in `\boxed{...}` (a pure numeric value or an expression reducible to an equivalent form).
- Do not output multiple candidate answers; do not describe the answer region in natural language.

Both training and evaluation enforce this format. The constraint itself constitutes part of the training signal: if the output violates the format, it receives no positive reward even if the final numeric value is correct.

This “format-as-compliance; value-as-correctness” standard is stricter than common looser evaluations (e.g., allowing free-form textual answers or minor formatting noise). It also explains why the same base model (Qwen2.5‑Math‑1.5B) achieves only about 5.4% pass@1 under our strict zero-shot setting, as opposed to often higher few-shot numbers reported elsewhere.

---

## 2. RLVR Pipeline (Reinforcement Learning with Verified Reward)

The full RL loop is closed and consists of the following steps:

### 2.1 Construct training samples and rollouts

- Use `get_gsm8k_train_ready_prompts()` to load and prepare GSM8K problems and gold answers into zero-shot prompt/answer pairs.
- Use `vLLM` to generate batched, parallel samples. Within each batch, the same problem produces multiple candidate answers (e.g., `group_size = 8`), each containing a full `<think> ... </think> <answer> ... </answer>` reasoning trace.
- `SamplingParams` controls temperature, max tokens, stop tokens, random seeds, etc., ensuring comparability within a group.

> vLLM is initialized with explicit settings (e.g., disabling FlashInfer, choosing a FlashAttention strategy, setting multiprocessing policy) to maintain consistent behavior across GPUs.

### 2.2 Verified Reward

- For each candidate answer, compute the reward using `drgrpo_grader.r1_zero_reward_fn`.
- The function:
  1) checks that the output strictly adheres to `<think> ... </think> <answer>\boxed{...}</answer>`;
  2) extracts the final answer from within `\boxed{...}`;
  3) compares the extracted answer with the gold using symbolic/numeric equivalence and algebraic simplification checks (adapted from `sail-sg/understand-r1-zero`’s `math_grader.py`), returning `reward ∈ {0.0, 1.0}`.

- Reward definition is simple and direct:
  - format-compliant and answer-correct → `reward = 1.0`
  - otherwise → `reward = 0.0`

- No human annotation, no extra reward model, no teacher logits distillation. The reward carries only one bit (correct/incorrect), but it is stable and fully automatable.

### 2.3 Advantage computation

- Answers to the same problem form a “group”.
- Compute the per-group baseline as the mean reward: $baseline：b = mean(reward_i)$.
- Each sample receives an advantage value

$$
 A_i = reward_i - b
$$

Differences from common RLHF practice:

- No standard deviation normalization (we do not divide by the group std). Std-normalization in small batches can amplify outliers and destabilize training by overfitting rare, very hard examples.
- Each sample is treated equally without additional weighting by token count or reasoning length to avoid the implicit bias “longer explanation = larger gradient.”

These advantage values are logged to Weights & Biases (e.g., `raw_reward/mean`, `advantage/mean`, `advantage/max`) to monitor the reward distribution.

### 2.4 Policy update

- The sampled answers (from the old policy) are re-scored by the current policy to compute token-level log-probabilities, which are then used to compute the loss and backpropagate updates.
- To control memory:
  - Use `bitsandbytes`’s `PagedAdamW8bit` optimizer.
  - Use `bfloat16`.
  - Use gradient accumulation (e.g., `gradient_accumulation_steps = 32`) to emulate a larger effective batch size (e.g., `train_batch_size = 64`) on a single 24GB GPU.
- If the advantage of a micro-batch is nearly zero, the update for that micro-batch can be skipped to avoid noise-dominated gradients.

### 2.5 Scheduling and evaluation

- After each `train_step`, the system:
  - logs token-level log-probabilities and entropy;
  - logs training status (e.g., `raw_reward/*`, `advantage/*`, `loss`, learning rate) to wandb.
- Every fixed number of steps (e.g., every 3 steps) call `grpo_evaluate.evaluate` to run a full zero-shot evaluation on the GSM8K test split and report pass@1 accuracy.
- Periodically (e.g., every 30 steps) save checkpoints (model, optimizer, scheduler, RNG states, wandb offset, etc.) and allow resuming from a specified `start_train_step`.

> The entire training–evaluation–logging loop is orchestrated by `cs336_alignment/grpo_with_checkpoint.py`.

---

## 3. Policy gradient objectives: REINFORCE+baseline and GRPO / Dr GRPO

We implement and compare three update rules:

- classical REINFORCE+baseline (no ratio clipping),
- original GRPO,
- Dr GRPO (the variant we adopt in the end).

All three use the same 0/1 verified reward; they differ in how gradients are constrained and whether additional per-sample scaling is applied.

### 3.1 REINFORCE + baseline

For sample $ i $:

- $reward_i$: 0/1 reward from `r1_zero_reward_fn`
- $b$: mean reward of the group (same problem)
- $A_i = reward_i - b$

Define the objective (to be minimized) in the classical REINFORCE form:

$$
L_{\mathrm{REINFORCE}}(\theta) = - \mathbb{E}_{i}\left[ A_i \log \pi_{\theta}(i) \right]
$$

Intuition:

- If an answer is better than the group average (typically “the correct one”), then $ A_i > 0 $ and optimization increases its probability.
- If it is worse than the average (typically “the incorrect ones”), then $ A_i < 0 $ and optimization decreases its probability.
- The baseline $ b $ reduces variance effectively, without training a separate value head.

Differences from common RLHF variants:

- No std-normalization of the advantage.
- No token-length weighting.
- No explicit KL penalty or entropy bonus to constrain policy drift; stability is maintained by within-batch comparison and learning rate control.

In practice, this variant can raise strict zero-shot pass@1 to about **87%** after roughly 200 on-policy update steps.

### 3.2 Original GRPO

Public GRPO (Group Relative Policy Optimization) can be viewed as adapting the idea of PPO-Clip to “multiple candidate answers for the same problem.” Let the $ i $-th complete answer be $ o_i $, its $ t $-th token be $ o_{i,t} $, and the question be $ q $. The group consists of $ \{o_1, \ldots, o_G\} $, where $ G $ is the group size. The (negative) loss to minimize is

$$
\mathcal{L}_{\mathrm{GRPO\text{-}Clip}}(\theta)
:= - \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\min\bigl(
  r_{i,t}(\theta)\,\hat{A}_{i,t},\;
  \mathrm{clip}\bigl(r_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\hat{A}_{i,t}
\bigr)
$$

where

$$
r_{i,t}(\theta)
:= \frac{\pi_{\theta}\bigl(o_{i,t}\mid q,\,o_{i,\lt t}\bigr)}
       {\pi_{\theta_{\mathrm{old}}}\bigl(o_{i,t}\mid q,\,o_{i,\lt t}\bigr)}
$$

and

$$
\hat{A}_{i,t}
:= \frac{
  R(q, o_i) - \mathrm{mean}\!\bigl(\{ R(q, o_1), \ldots, R(q, o_G) \}\bigr)
}{
  \mathrm{std}\!\bigl(\{ R(q, o_1), \ldots, R(q, o_G) \}\bigr)
}.
$$

Two salient properties:

- Token-length weighting. The outer factor $ \frac{1}{|o_i|} $ averages gradients across the tokens of a long answer, effectively down-weighting long answers and up-weighting short ones at the sequence level.
- Std normalization. $ \hat{A}_{i,t} $ divides by the group standard deviation, akin to z-score normalization within the batch, which emphasizes relatively better answers while controlling variance.

### 3.3 Dr GRPO (a GRPO variant)

Dr GRPO follows GRPO’s spirit by using ratio clipping to control the magnitude of each update (in the PPO/PPO-Clip style), but removes the two bias-amplifying factors above. Let

- `old_log_prob_i`: the log-probability of answer $ i $ under the rollout (old) policy;
- `new_log_prob_i`: the log-probability of the same answer under the current policy;
- probability ratio

$$
  r_i = \exp(\log \pi_\theta(i) - \log \pi_{\text{old}}(i))
$$

We still use $ A_i = \mathrm{reward}_i - b $, as in REINFORCE, without std-normalization.

The clipped surrogate objective (to minimize) is

$$
L(\theta)
:= \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\bigl(
  \frac{\pi_{\theta}(o_{i,t} \mid q,\, o_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q,\, o_{i,\lt t})} \,\hat{A}_{i,t},
  \mathrm{clip}\bigl(
    \frac{\pi_{\theta}(o_{i,t} \mid q,\, o_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q,\, o_{i,\lt t})},
    1-\varepsilon,\,
    1+\varepsilon
  \bigr)\,\hat{A}_{i,t}
\bigr)
$$

$$
\hat{A}_{i,t}
:= R(q, o_i) - \mathrm{mean}\bigl( R(q, o_1), \ldots, R(q, o_G) \bigr)
$$

The clipping range $ \varepsilon $ is tunable (we use `0.30`). Intuition:

- When the new policy attempts to “over-boost” the probability of an answer (i.e., $ r_i $ deviates too far from 1), clipping limits the update magnitude to avoid overfitting in one step.

Key differences from GRPO:

- We no longer apply the $ \frac{1}{|o_i|} $ token-length weighting to re-scale all tokens within a long answer, which mitigates the tendency to produce ever-longer answers.
- We do not divide by the group standard deviation. Instead, $ \hat{A}_{i,t} = \mathrm{reward}_i - \mathrm{mean\_reward} $. This avoids over-amplifying rare high-scoring answers when the batch std is very small in small-batch settings, and avoids artificially stretching a discrete 0/1 reward.

Training details:

- Perform multiple epochs over the same rollout batch (e.g., `epoch_per_rollout_batch = 3`).
- Use the unclipped term in the first epoch; in subsequent epochs, continue updating on the same samples with ratio clipping, steadily absorbing the batch signal without introducing new samples.

Empirical results:

- Dr GRPO reaches about **83%** pass@1 within a comparable number of steps.
- Although slightly lower than REINFORCE+baseline (~87%), Dr GRPO trains more smoothly when scaling batch size (e.g., from ~1.6k to ~6.4k samples) and migrating to higher-bandwidth GPUs (H200), offering better stability and scalability.
- As with REINFORCE, we do not add KL penalties or entropy bonuses; stability comes from clipping and modest learning rates.

### 3.4 Ablations: why we ultimately adopt Dr GRPO

We conduct four ablations on GRPO-style objectives, toggling different scaling factors to examine stability and final accuracy:

- Original GRPO: both token-length weighting ($1/|o_i|$) and std normalization are enabled.
- GRPO without std normalization: remove std from $ \hat{A}_{i,t} $ while retaining token-length weighting.
- GRPO without token-length weighting: remove $1/|o_i|$ but keep z-score style $ (\mathrm{reward}_i - \mathrm{mean})/\mathrm{std} $.
- Dr GRPO (remove both): no $1/|o_i|$ and no std normalization; i.e., the formula in §3.3.

We select Dr GRPO for further experiments because it better suppresses answer-length inflation and yields steadier training without large gradient spikes across batches caused by a few high-reward answers, thus utilizing the reward signal more effectively.

In combination with §3.1’s REINFORCE+baseline (no clipping; larger single-step updates), we retain two training paths:

- REINFORCE+baseline: reaches ~87% quickly under small compute/small batches.
- Dr GRPO: maintains stability under larger compute and higher-throughput rollouts, reaching ~83% and scaling more gracefully.

### 3.5 Summary

- Both methods use the exact same automatically verifiable reward.
- REINFORCE+baseline climbs to ~87% within ~200 steps.
- Dr GRPO provides a PPO/GRPO-style update with stable training at larger throughput, reaching ~83%.

This demonstrates that for a 1.5B-parameter mathematical reasoning model, verified reward plus policy gradient alone (no RM, no teacher) can substantially improve reasoning ability.

---

## 4. Resource efficiency and scalability

The system is optimized for low-resource RL while supporting migration to high-bandwidth GPUs.

- **Single NVIDIA L4 (24GB)**
  - Development, tuning, and stability validation were performed primarily on a single L4 (24GB), totaling over 200 GPU hours.
  - With `bfloat16`, `PagedAdamW8bit`, gradient accumulation, and decoupling rollout from backpropagation, RL runs stably within 24GB.
  - Full integration with `wandb` for monitoring reward distribution, advantage distribution, token-level entropy, loss curves, and LR scheduling.

- **H200 scaling**
  - The same code and logic transfer directly to NVIDIA H200, expanding single-rollout size from ~1.6k to ~6.4k samples (~50 GPU hours).
  - Similar numbers of update steps and comparable convergence levels are preserved, indicating near-linear throughput scaling without algorithm redesign.

- **Hyperparameter adaptation**
  - Adjust `gpu_memory_utilization`, `sampling_max_tokens`, `rollout_batch_size`, etc., to fit different memory budgets.
  - Typical settings: `gpu_memory_utilization = 0.80`, `sampling_max_tokens = 512`, `gradient_accumulation_steps = 32`, `epoch_per_rollout_batch = 3`.

---

## 5.1 Experimental results (GSM8K test split, strict zero-shot)

| Training strategy / model                 | pass@1 Accuracy | Training steps           | Notes                                               |
| ---------------------------------------- | --------------- | ------------------------ | --------------------------------------------------- |
| Qwen2.5‑Math‑1.5B (base)                 | ~5.4%           | –                        | Strict zero-shot baseline                           |
| REINFORCE+baseline (this system)         | ~87%            | ~200 on-policy steps     | No KL/entropy regularizers; relies on group baseline |
| Dr GRPO / clipped GRPO (this system)     | ~83%            | ~200 steps               | Ratio clipping (clip≈0.30); more stable, slightly lower |
| Qwen2.5-Math-7B (base)       | ~6.1%           | -                   | Measured under the same evaluation script                 |
| Llama‑3.1‑8B‑Instruct                     | ~41.3%          | –                        | Measured under the same evaluation script           |
| Qwen3‑8B (thinking mode)                  | ~14.9%          | –                        | Measured under the same evaluation script           |

Notes:

- All models are evaluated under the same script and strict criteria.
- The evaluation requires both the correct format and the correct numeric answer; free-form textual answers or “nearly correct” outputs are counted as incorrect.
- Therefore, the reported accuracies are directly comparable under identical conditions.

---


## 5.2 Experimental results（GSM8K test split, Non-Strict Parser 0-shot）

| Training strategy / model                 | pass@1 Accuracy | Training steps           | Notes                                               |
| ---------------------------- | --------------- | ------------------- | -------------------------------- |
| Qwen2.5-Math-1.5B (base)       | ~30.5%           | -                   | 0-shot baseline                     |
| REINFORCE+baseline (this system)     | ~89%            | ~200 on-policy step | No KL/entropy regularizers; relies on group baseline |
| Dr GRPO / clipped GRPO (this system) | ~84%            | ~200 step           | Ratio clipping (clip≈0.30); more stable, slightly lower |
| Qwen2.5-Math-7B (base)       | ~64.7%           | -                   |  Measured under the same evaluation script                    |
| Llama-3.1-8B-Instruct        | ~48.9%          | -                   |  Measured under the same evaluation script                          |
| Qwen3-8B (thinking mode)       | ~82.2%          | -                   |  Measured under the same evaluation script                       |

Notes：

* All models are evaluated under the same script and Non-strict criteria.
* Natural language answer and Non-JSON answer are allowed

---


## 5.3 Experimental results (MATH500, strict zero-shot)

| Training strategy / model                 | pass@1 Accuracy | Training steps           | Notes                                               |
| ---------------------------------------- | --------------- | ------------------------ | --------------------------------------------------- |
| Qwen2.5‑Math‑1.5B (base)                 | ~4.6%           | –                        | Strict zero-shot baseline                           |
| REINFORCE+baseline (this system)         | ~67.8%            | ~200 on-policy steps     | No KL/entropy regularizers; relies on group baseline |
| Qwen2.5-Math-7B (base)       | ~2.8%           | -                   | Measured under the same evaluation script                 |
| Llama‑3.1‑8B‑Instruct                     | ~14.4%          | –                        | Measured under the same evaluation script           |
| Qwen3‑8B (thinking mode)                  | ~21.4%          | –                        | Measured under the same evaluation script           |

Notes:

- All models are evaluated under the same script and strict criteria.
- The evaluation requires both the correct format and the correct numeric answer; free-form textual answers or “nearly correct” outputs are counted as incorrect.
- Therefore, the reported accuracies are directly comparable under identical conditions.

---



## 5.4 Experimental results（MATH500, Non-Strict Parser 0-shot）

| Training strategy / model                 | pass@1 Accuracy | Training steps           | Notes                                               |
| ---------------------------- | --------------- | ------------------- | -------------------------------- |
| Qwen2.5-Math-1.5B (base)       | ~23.2%           | -                   | 0-shot baseline                     |
| REINFORCE+baseline (this system)     | ~69.6%            | ~200 on-policy step | No KL/entropy regularizers; relies on group baseline |
| Qwen2.5-Math-7B (base)       | ~57.2%           | -                   |  Measured under the same evaluation script                    |
| Llama-3.1-8B-Instruct        | ~22.8%          | -                   |  Measured under the same evaluation script                          |
| Qwen3-8B (thinking mode)       | ~67.4%          | -                   |  Measured under the same evaluation script                       |

Notes：

* All models are evaluated under the same script and Non-strict criteria.
* Natural language answer and Non-JSON answer are allowed



## 6. Training monitoring and visualization

All key metrics are automatically logged to Weights & Biases (wandb), including but not limited to:

- `raw_reward/*`: mean reward per batch, max reward, etc.
- `advantage/*`: group advantage distribution and marginal comparisons.
- `token/*`: token-level log probability and token-level entropy (to monitor mode collapse or overconfidence).
- `eval_metrics/*`: periodic pass@1 on the GSM8K test split.
- `optimizer/*`: learning rate, gradient norms, etc.

Token entropy and evaluation accuracy curves are shown below:

![Token entropy](./figure/tokenentropy.png)
*Figure 1. Token-level entropy vs. training steps, used to monitor whether the model collapses prematurely into a fixed response template.*

![Evaluation accuracy](./figure/eval.png)
*Figure 2. Evaluation accuracy vs. training steps. Accuracy rises rapidly within ~200 steps and then stabilizes.*

---

## 7. Checkpointing and reproducibility

The script `cs336_alignment/grpo_with_checkpoint.py` supports full checkpointing for long training runs and cross-GPU migration.

- `save_checkpoint()` stores:
  - model weights
  - optimizer state
  - LR scheduler state
  - RNG states (`random` / `numpy` / `torch` / `torch.cuda`)
  - current global step counter
  - wandb run offset

- By setting `Load_From_Checkpoint=True` and specifying `checkpoint_dir` and `start_train_step`, training can resume on another GPU without re-warmup.
- Random seeds are fixed/restored explicitly, and non-deterministic CuDNN paths are disabled to improve experimental reproducibility across sessions.

---

## 8. Reproduction steps

1) Install dependencies

```bash
uv sync
```

2) Prepare model weights

- Pre-download `Qwen/Qwen2.5-Math-1.5B` to the local cache.
- Optionally set `HF_HUB_OFFLINE=1` at runtime to avoid network-induced interruptions.

3) Start training

```bash
uv run python cs336_alignment/grpo_with_checkpoint.py
```

4) Monitor training

- Use wandb to monitor reward, advantage, token entropy, and evaluation accuracy in real time.
- If memory is limited, lower `rollout_batch_size` and `sampling_max_tokens`, or increase `gradient_accumulation_steps`.

5) Evaluate

- Evaluation logs are written to `./outputs/`.
- The terminal prints pass@1 on the GSM8K test split for direct comparison against the baselines above.

> Checkpoints are saved periodically for seamless recovery after interruptions.

---

## 9. Code structure

```text
.
├── cs336_alignment/
│   ├── grpo_with_checkpoint.py   # RL entry point (rollout / train / eval / checkpoint)
│   ├── grpo_utility.py           # Core helpers: policy update, advantage, log prob / entropy
│   └── ...                       # Hyperparameters and scheduling logic
├── drgrpo_grader/                # Verified reward and format checking (adapted from math_grader)
├── outputs/                      # Evaluation logs and metric snapshots
├── grpo_checkpoint/              # Checkpoints (model, optimizer, RNG, etc.)
└── figure/                       # Visualization (token entropy / eval accuracy)
```

- `grpo_with_checkpoint.py`: drives rollout → scoring → advantage → policy update → periodic evaluation → checkpointing.
- `grpo_utility.py`: core RL logic including REINFORCE+baseline, Dr GRPO, ratio clipping, log-prob and entropy extraction, and gradient accumulation control.
- `drgrpo_grader/`: contains `r1_zero_reward_fn` and math answer parsing/verification logic (adapted from `sail-sg/understand-r1-zero`’s `math_grader.py` to fit the project’s output format constraint).
- `outputs/`: training and evaluation results (including pass@1 accuracy).
- `grpo_checkpoint/`: portable recovery points.

---

## 10. Future directions

- Finer-grained reward signals. Current reward is binary (correct=1 / incorrect=0). Future work may decompose reward into “format correctness + numeric correctness + step consistency,” increasing signal density without sacrificing full automation.
- Adaptive training strategies. Use token-level entropy and evaluation trajectories to adjust LR, clip range, and `group_size` dynamically, balancing exploration and convergence across training phases.
- Cross-dataset generalization. Directly transfer the same pipeline to harder reasoning benchmarks (AIME24, MATH500, etc.) to test out-of-distribution performance.
- Multi-GPU / larger-batch rollouts. Extend vLLM rollouts to multi-GPU parallelism; combine with activation checkpointing and pipelined sampling–updating to further improve throughput and sample efficiency.

---

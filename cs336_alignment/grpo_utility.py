import os
from typing import Any, Callable, Literal

from huggingface_hub.inference._generated.types.text_to_speech import TextToSpeechEarlyStoppingEnum
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
import json
import re
from string import Template
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from transformers import AutoModelForCausalLM, PreTrainedModel

def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rollout_len = len(rollout_responses)
    raw_rewards_per_group = torch.empty(group_size)
    raw_reward = torch.empty(rollout_len)
    average_reward_per_group = torch.empty(int(rollout_len/group_size))
    std_reward_per_group = torch.empty(int(rollout_len/group_size))

    advantage = torch.empty(rollout_len)
    batch_index = 0
    for i,x in enumerate(rollout_responses):
        reward = torch.tensor(reward_fn(x, repeated_ground_truths[i])["reward"])
        raw_reward[i] = reward
        raw_rewards_per_group[int(i%group_size)] = reward
        if (i+1) % group_size == 0:
            average_reward = torch.mean(raw_rewards_per_group)
            average_reward_per_group[batch_index] = average_reward
            std_reward = torch.std(raw_rewards_per_group)
            std_reward_per_group[batch_index] = std_reward
            batch_index += 1
            raw_rewards_per_group = torch.empty(group_size)
    
    batch_index = 0

    for i, x in enumerate(raw_reward):
        if (i) % group_size == 0:
            average_reward = average_reward_per_group[batch_index]
            if normalize_by_std:
                std_reward = std_reward_per_group[batch_index]
            batch_index += 1
        if normalize_by_std:
            advantage[i] = (x - average_reward) / (std_reward + advantage_eps)
        else:
            advantage[i] = x - average_reward

    return (advantage, raw_reward, {})





def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -(raw_rewards_or_advantages * policy_log_probs)




def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip = torch.clamp(ratio, 1-cliprange, 1+cliprange)
    grpo_clip_loss = -torch.min(advantages * clip, advantages * ratio)
    return grpo_clip_loss, {}



def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    elif loss_type == "grpo_clip":
        assert old_log_probs is not None and cliprange is not None and advantages is not None
        loss, _= compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss, {}



def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    else:
        return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)



def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    # batch_size = policy_log_probs.shape[0]
    loss, _ = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask)
    loss = loss / float(gradient_accumulation_steps)
    loss.backward()
    return loss, {}



def get_gsm8k_train_ready_prompts() -> tuple[list[str], list[str]]:
    prompts = Template(Path("cs336_alignment/prompts/r1_zero_inference.prompt").read_text(encoding="utf-8"))
    rendered_prompts, answers_pure = [], []
    with open("data/gsm8k/train_grpo_L4.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            answer = example["answer"]
            matches = re.findall(r"####\s*([^\n]+)", answer)
            if matches:
                matches = matches[0].strip()
                answers_pure.append(matches)
            else:
                # skip if no extact answer match
                continue
            rendered_prompts.append(prompts.substitute(question=example["question"]))
    return rendered_prompts, answers_pure




def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.80):
    vllm_set_random_seed(seed)

    ## 防止vllm环境报错
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True
    )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def rollout_vllm(vllm: LLM, prompts: list[str], sampling_params: SamplingParams) -> tuple[list[str], list[list[float]], list[list[int]]]:
    raw_responses = vllm.generate(prompts, sampling_params)
    texts, log_probs, token_ids = [], [], []
    ## 外层为每条prompt的多个response
    for line in raw_responses:
        ## 内层为每条prompt的多个response
        for output in line.outputs:
            texts.append(output.text.strip())
            log_probs.append(output.logprobs)
            token_ids.append(output.token_ids)
    return texts, log_probs, token_ids


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    return -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids, attention_mask=attn_mask).logits[:, :-1, :]
    log_prob = torch.log_softmax(logits, dim=-1)
    log_prob = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": log_prob, "token_entropy": token_entropy}
    else:
        return {"log_probs": log_prob}



def get_response_log_probs_tensor_and_response_mask(
    new_log_probs: torch.Tensor,
    expanded_old_log_probs: list[list[float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs_len = [len(x) for x in expanded_old_log_probs]
    response_len = log_probs_len
    max_response_len = max(response_len)


    old_log_probs_tensor = torch.zeros(
        len(expanded_old_log_probs),
        max_response_len,
        dtype=torch.bfloat16,
        device="cuda"
    )

    response_mask = torch.zeros_like(old_log_probs_tensor, dtype=torch.bfloat16, device="cuda")
    

    for i, (seq, resp_len) in enumerate(zip(expanded_old_log_probs, response_len)):
        old_log_probs_tensor[i, -resp_len:] = torch.tensor(seq, dtype=torch.bfloat16, device="cuda")
        response_mask[i, -resp_len:] = 1.0
    
    new_lpg_probs_resp = torch.zeros_like(old_log_probs_tensor)
    for i, resp_len in enumerate(response_len):
        new_lpg_probs_resp[i, -resp_len:] = new_log_probs[i, -resp_len:]
    
    return old_log_probs_tensor, response_mask, new_lpg_probs_resp

## 需要修改
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: SamplingParams,
    ground_truths: list[str]
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    raw_responses = vllm_model.generate(prompts, eval_sampling_params)
    responses = []

    for line in raw_responses:
        response = line.outputs[0].text.strip()
        responses.append(response)

    all_metrics = []
    correct_count = 0
    
    for i, (prompt, response, ground_truth) in enumerate(zip(prompts, responses, ground_truths), start=1):

        metrics = reward_fn(response, ground_truth)
        metrics["index"] = i
        correct_count += metrics["answer_reward"]
        metrics["prompt"] = prompt
        metrics["response"] = response
        metrics["ground_truth"] = ground_truth
        all_metrics.append(metrics)

    print(f"Correct count: {correct_count}")
    print(f"Accuracy: {correct_count / len(prompts)}")

    ## write to a jsonl file
    with open("outputs/expert2_sft_metrics.jsonl", "w") as f:
        for metrics in all_metrics:
            f.write(json.dumps(metrics) + "\n")
    print(f"Saved metrics to expert2_sft_metrics.jsonl")
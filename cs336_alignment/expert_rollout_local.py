from vllm import LLM, SamplingParams
import drgrpo_grader as grader
from pathlib import Path
import json
import re
from statistics import mean, median
from transformers import AutoTokenizer
from typing import Callable
from string import Template
from unittest.mock import patch
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed





prompts = Template(Path("cs336_alignment/prompts/r1_zero_inference.prompt").read_text(encoding="utf-8"))

questions, answers, rendered_prompts, answers_pure = [], [], [], []

with open("data/gsm8k/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        questions.append(example["question"])
        answers.append(example["answer"])
        matches = re.findall(r"####\s*([^\n]+)", answers[-1])
        if matches:
            matches = matches[0].strip()
            answers_pure.append(matches)
        else:
            # attach the empty string if no matches
            answers_pure.append("")
        rendered_prompts.append(prompts.substitute(question=example["question"]))


sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=1024,
    min_tokens=4,
    stop=["</answer>"],
    include_stop_str_in_output=True,
    n=2,
    seed=99)


# llm = LLM(
#         model="Qwen/Qwen2.5-Math-1.5B",
#         dtype="bfloat16")
#         # max_num_batched_tokens=8000)


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
        response = line.outputs
        responses.append(response)

    all_metrics = []
    correct_count_first = 0
    correct_count_second = 0
    correct_count = 0
    
    for i, (prompt, response, ground_truth) in enumerate(zip(prompts, responses, ground_truths), start=1):
        is_correct = False
        for j, output in enumerate(response, start=1):
            output = output.text.strip()

            metrics = reward_fn(output, ground_truth)
            metrics["index"] = i
            metrics["n"] = j
            if metrics["answer_reward"] == 1.0:
                is_correct = True
            if j == 1:
                correct_count_first += metrics["answer_reward"]
            else:
                correct_count_second += metrics["answer_reward"]
            metrics["prompt"] = prompt
            metrics["response"] = output
            metrics["ground_truth"] = ground_truth
            all_metrics.append(metrics)
        if is_correct:
            correct_count += 1

    print(f"Correct count first: {correct_count_first}")
    print(f"Correct count second: {correct_count_second}")
    print(f"Correct count: {correct_count}")
    print(f"Accuracy first: {correct_count_first / len(prompts)}")
    print(f"Accuracy second: {correct_count_second / len(prompts)}")
    print(f"Accuracy total: {correct_count / len(prompts)}")
    ## write to a jsonl file
    with open("outputs/expert_rollout_4.jsonl", "w") as f:
        for metrics in all_metrics:
            f.write(json.dumps(metrics) + "\n")
    print(f"Saved metrics to expert_rollout_4.jsonl")
    
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.90):
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
        gpu_memory_utilization=gpu_memory_utilization
    )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

## 不需要传到cuda, 浪费显存，只需要传入dict
policy = AutoModelForCausalLM.from_pretrained("models/expert_3",
                                            torch_dtype="bfloat16",
                                            attn_implementation="flash_attention_2")
llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", "cuda", 99)
load_policy_into_vllm_instance(policy, llm)

evaluate_vllm(llm, grader.r1_zero_reward_fn, rendered_prompts, sampling_params, answers_pure)

    

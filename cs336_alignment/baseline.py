from vllm import LLM, SamplingParams
import drgrpo_grader as grader
from pathlib import Path
import json
import re
from statistics import mean, median
from transformers import AutoTokenizer
from typing import Callable






prompts = Path("cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8")

questions, answers, rendered_prompts, answers_pure = [], [], [], []

with open("data/gsm8k/test.jsonl", "r", encoding="utf-8") as f:
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
        rendered_prompts.append(prompts.format(question=example["question"]))
        # if len(questions) >= 2:
        #     break

# print(questions)
# print(answers)
# print(rendered_prompts)
# print(answers_pure)


# TOKENIZER_NAME = "Qwen/Qwen2.5-Math-1.5B"

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# def count_tokens(s: str) -> int:
#     # 不加 special tokens，只统计纯文本token
#     return len(tokenizer.encode(s, add_special_tokens=False))

# def prompt_length_stats_by_tokens(rendered_prompts: list[str]) -> dict:
#     if not rendered_prompts:
#         return {"count": 0, "min": 0, "median": 0, "mean": 0.0, "max": 0}

#     lens = [count_tokens(s) for s in rendered_prompts]
#     return {
#         "count": len(lens),
#         "min": min(lens),
#         "median": int(median(lens)),
#         "mean": float(mean(lens)),
#         "max": max(lens)
#     }


# print(prompt_length_stats_by_tokens(rendered_prompts))






# # print(grader.grade(answers[0], answers_pure[0], fast=True))


sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True)


llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        dtype="bfloat16")
        # max_num_batched_tokens=8000)


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
        metrics = {}
        metrics["index"] = i
        metrics = reward_fn(response, ground_truth)
        correct_count += metrics["answer_reward"]
        metrics["prompt"] = prompt
        metrics["response"] = response
        metrics["ground_truth"] = ground_truth
        all_metrics.append(metrics)

    print(f"Correct count: {correct_count}")
    print(f"Accuracy: {correct_count / len(prompts)}")

    ## write to a jsonl file
    with open("outputs/zero_shot_metrics.jsonl", "w") as f:
        for metrics in all_metrics:
            f.write(json.dumps(metrics) + "\n")
    print(f"Saved metrics to zero_shot_metrics.jsonl")
    


evaluate_vllm(llm, grader.r1_zero_reward_fn, rendered_prompts, sampling_params, answers_pure)

    
    



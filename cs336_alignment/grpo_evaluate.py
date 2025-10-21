from vllm import LLM, SamplingParams
from pathlib import Path
import json
import re
from typing import Callable
from string import Template


def evaluate(LLM: LLM, model_name: str, reward_fn: Callable[[str, str], dict[str, float]], max_tokens: int = 512):

    ROOT = Path(__file__).resolve().parents[1]

    template_path = ROOT / "cs336_alignment" / "prompts" / "r1_zero_inference.prompt"

    prompts = Template(template_path.read_text(encoding="utf-8"))

    questions, answers, rendered_prompts, answers_pure = [], [], [], []
    sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=max_tokens,
    min_tokens=4,
    stop=["</answer>"],
    include_stop_str_in_output=True)

    with open(ROOT / "data" / "gsm8k" / "test.jsonl", "r", encoding="utf-8") as f:
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


    def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: list[str],
        eval_sampling_params: SamplingParams,
        ground_truths: list[str]
    ) -> tuple[int, float]:
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
        accuracy = correct_count / len(prompts)
        print(f"Correct count: {correct_count}")
        print(f"Accuracy: {accuracy}")

        ## write to a jsonl file
        with open(f"outputs/grpo/{model_name}_metrics.jsonl", "w") as f:
            for metrics in all_metrics:
                f.write(json.dumps(metrics) + "\n")
        print(f"Saved metrics to {model_name}_metrics.jsonl")
        return correct_count, accuracy

    correct_count, accuracy = evaluate_vllm(LLM, reward_fn, rendered_prompts, sampling_params, answers_pure)
    return correct_count, accuracy



def evaluate_vllm_with_multiple(LLM: LLM, model_name: str, reward_fn: Callable[[str, str], dict[str, float]], max_tokens: int = 512):

    ROOT = Path(__file__).resolve().parents[1]

    template_path = ROOT / "cs336_alignment" / "prompts" / "r1_zero_inference.prompt"

    prompts = Template(template_path.read_text(encoding="utf-8"))

    questions, answers, rendered_prompts, answers_pure = [], [], [], []
    sampling_params = SamplingParams(
    n=8,
    temperature=1.0,
    top_p=1.0,
    max_tokens=max_tokens,
    min_tokens=4,
    stop=["</answer>"],
    include_stop_str_in_output=True)

    with open(ROOT / "data" / "gsm8k" / "test.jsonl", "r", encoding="utf-8") as f:
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


    def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: list[str],
        eval_sampling_params: SamplingParams,
        ground_truths: list[str]
    ) -> tuple[int, float]:
        """
        Evaluate a language model on a list of prompts,
        compute evaluation metrics, and serialize results to disk.
        """
        raw_responses = vllm_model.generate(prompts, eval_sampling_params)
        responses = []
        for prompt_response in raw_responses:
            for output in prompt_response.outputs:
                response = output.text.strip()
                responses.append(response)

        all_metrics = []
        correct_count = 0

        repeated_ground_truths = [ground_truth for ground_truth in ground_truths for _ in range(8)]
        repeated_prompts = [prompt for prompt in prompts for _ in range(8)]
        rewards = []
        for i, (prompt, response, ground_truth) in enumerate(zip(repeated_prompts, responses, repeated_ground_truths), start=1):

            metrics = reward_fn(response, ground_truth)
            metrics["index"] = i
            rewards.append(metrics["answer_reward"])
            correct_count += metrics["answer_reward"]
            metrics["prompt"] = prompt
            metrics["response"] = response
            metrics["ground_truth"] = ground_truth
            all_metrics.append(metrics)
        
        correct_fused = 0
        is_correct = False
        for i, reward in enumerate(rewards):
            if reward == 1.0 and is_correct == False:
                is_correct = True
                correct_fused += 1
                continue
            if (i+1) % 8 == 0:
                is_correct = False
        
        print(f"Correct count fused: {correct_fused}")
        print(f"Accuracy fused: {correct_fused / len(prompts)}")
        # accuracy = correct_count / len(prompts)
        # print(f"Correct count: {correct_count}")
        # print(f"Accuracy: {accuracy}")

        ## write to a jsonl file
        # with open(f"outputs/grpo/{model_name}_metrics.jsonl", "w") as f:
        #     for metrics in all_metrics:
        #         f.write(json.dumps(metrics) + "\n")
        # print(f"Saved metrics to {model_name}_metrics.jsonl")
        # return correct_count, accuracy

    evaluate_vllm(LLM, reward_fn, rendered_prompts, sampling_params, answers_pure)
    # return correct_count, accuracy



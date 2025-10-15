import os
# # 使用v1版本, 拿到原始softmax的分布而不是topk等调整后的分布
# # 注意，这里仍存在问题，因为rollout的分布是topk/temperature调整后的分布，而不是原始softmax的分布
# # 这可能会导致轻微的“off policy”问题，但目前看来影响不大
# os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from grpo_utility import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
# from sft_evaluate import init_vllm, load_policy_into_vllm_instance, evaluate_vllm
from torch.utils.data import DataLoader
from drgrpo_grader import r1_zero_reward_fn


import time

time_start = time.time()

n_grpo_steps = 200
learning_rate = 1e-5
advantage_eps = 1e-8
sampling_min_tokens = 4
sampling_max_tokens = 512
epoch_per_rollout_batch = 1
train_batch_size = 64
rollout_batch_size = 64
gradient_accumulation_steps = 64
gpu_memory_utilization = 0.80
loss_type = "reinforce_with_baseline"
group_size = 8

use_std_normalization = True


def main():
    # policy = AutoModelForCausalLM.from_pretrained(
    #     "Qwen/Qwen2.5-Math-1.5B",
    #     torch_dtype="bfloat16",
    #     attn_implementation="flash_attention_2"
    # )
    # optimizer = torch.optim.Adam(policy.parameters(),
    #                             lr=learning_rate,
    #                             betas=(0.9, 0.95),
    #                             weight_decay=0.0)


    ## 64 // 64 == 0
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    ## 64/ 64 = 1
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    ## 64 // 8 == 0
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    ## 64/ 8 = 8
    n_prompt_per_rollout_batch = rollout_batch_size // group_size

    ## 64 >= 8
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    ## 64 // 1 == 64
    n_micro_batches_per_epoch = rollout_batch_size // micro_train_batch_size

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,
        seed=99,
        logprobs=1
    )

    i, j = 0, 0
    prompts, answers_pure = get_gsm8k_train_ready_prompts()
    dataset = list(zip(prompts, answers_pure))
    ## 8 prompts per batch
    dataloader = DataLoader(
        dataset,
        batch_size=n_prompt_per_rollout_batch,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    dataloader_iter = iter(dataloader)


    for train_step in range(n_grpo_steps):
        batch = next(dataloader_iter)
        prompts, answers_pure = batch
        if train_step == 0:
            policy = AutoModelForCausalLM.from_pretrained(
                # f"Model/policy_{train_step}",
                "Qwen/Qwen2.5-Math-1.5B",
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2"
            )
            llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
        else:
            # policy = AutoModelForCausalLM.from_pretrained(
            #     ## 如果train_batch_size=rollout_batch_size, 则不适用grpo, 使用reinforcewithbaseline, 每次rollout只take one otimp step.
            #     f"Model/policy_{train_step}",
            #     torch_dtype="bfloat16",
            #     attn_implementation="flash_attention_2"
            # )
            llm = init_vllm("Qwen/Qwen2.5-Math-1.5B", "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
        
        ## 现在rollout, 8 prompts得到8*8=64个response
        texts, log_probs, token_ids = rollout_vllm(llm, prompts, sampling_params)
        correct_count = 0
        all_metrics = []
        answers_repeated = [x for x in answers_pure for _ in range(group_size)]
        for i, (prompt, response, ground_truth) in enumerate(zip(prompts, texts, answers_repeated), start=1):
            metrics = r1_zero_reward_fn(response, ground_truth)
            metrics["index"] = i
            correct_count += metrics["answer_reward"]
            all_metrics.append(metrics)

        print(f"Correct count: {correct_count}")
        print(f"Accuracy: {correct_count / len(texts)}")
        print("length of texts is: ")
        print(len(texts))
        print("length of prompts is: ")
        print(len(prompts))
        print("length of answers_pure is: ")
        print(len(answers_pure))
        print("length of answers_repeated is: ")
        print(len(answers_repeated))
        print("reard is :")
        print(all_metrics)
        print("-"*100)
        print("texts is: ")
        print(texts[0:8])
        print("-"*100)
        print("answers_pure is: ")
        print(answers_pure[0])
        print("-"*100)
        print("log_probs is: ")
        print(log_probs[0])
        print("-"*100)
        print("token_ids is: ")
        print(token_ids[0])
        print("-"*100)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        exit()

        answers_repeated = [x for x in answers_pure for _ in range(group_size)]
        # reponse_list_by_group = [raw_responses[i:i+group_size] for i in range(0, len(raw_responses), group_size)]
        ## 32
        for i in range(gradient_accumulation_steps):
            advantages, _, _ = compute_group_normalized_rewards(r1_zero_reward_fn, outputs, answers_repeated, group_size, advantage_eps, use_std_normalization)
            
            
            

        


if __name__ == "__main__":
    main()


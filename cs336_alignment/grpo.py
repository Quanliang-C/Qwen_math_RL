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
import gc

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
        texts, old_log_probs, token_ids = rollout_vllm(llm, prompts, sampling_params)
        # print("first old_log_probs length", len(old_log_probs[0]))
        # old_log_probs = old_log_probs[0][0].get(token_ids[0][0])
        # print("old_log_probs", old_log_probs)
        # old_log_probs = [ for x, y in zip(old_log_probs, token_ids)]
        expanded_old_log_probs = []
        for step_logprobs, step_token_ids in zip(old_log_probs, token_ids):
            seq_log_probs = []
            for per_step_dict, tid in zip(step_logprobs, step_token_ids):
                if per_step_dict is None:
                    seq_log_probs.append(None)            # prompt 部分可能是 None
                else:
                    seq_log_probs.append(per_step_dict[tid].logprob)
            expanded_old_log_probs.append(seq_log_probs)

        old_log_probs = expanded_old_log_probs
        print("first old_log_probs length", len(old_log_probs[0]))
        print("first old_log_probs", old_log_probs[0])
        print("len of old log probs", len(old_log_probs))
        # print("old_log_probs", old_log_probs)
        


        response_len = [len(x) for x in texts]
        print("response_len", response_len)
        log_probs_len = [len(x) for x in old_log_probs]
        print("log_probs_len", log_probs_len)
        token_ids_len = [len(x) for x in token_ids]
        print("token_ids_len", token_ids_len)
        exit()





        correct_count = 0
        all_metrics = []
        answers_repeated = [x for x in answers_pure for _ in range(group_size)]
        prompt_repeated = [x for x in prompts for _ in range(group_size)]
        advantage, raw_reward, _ = compute_group_normalized_rewards(r1_zero_reward_fn, texts, answers_repeated, group_size, advantage_eps, use_std_normalization)
        prompt_and_response = [x + y for x, y in zip(prompt_repeated, texts)]
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        tokenizer.padding_side = "left"
        prompt_and_response_token_ids = tokenizer(
            prompt_and_response,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        del llm
        gc.collect()
        torch.cuda.empty_cache()


        input_ids = prompt_and_response_token_ids["input_ids"].to("cuda")
        attention_mask = prompt_and_response_token_ids["attention_mask"].to("cuda")
        # attention_mask = attention_mask[:, 1:]
        labels = input_ids[:, 1:].to("cuda")
        print("input_ids in", input_ids.device)
        print("attention_mask in", attention_mask.device)
        print("input_ids shape", input_ids.shape)
        print("attention_mask shape", attention_mask.shape)
        print("labels in", labels.device)
        print("labels shape", labels.shape)

        
        # new_log_probs = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False, attn_mask=attention_mask)
        # print("new_log_probs in", new_log_probs.device)
        # print("new_log_probs shape", new_log_probs.shape)
        
        policy.to("cuda")
        for i in range(epoch_per_rollout_batch):
            for j in range(0, rollout_batch_size, micro_train_batch_size):
                new_log_probs = get_response_log_probs(policy, input_ids[j:j+micro_train_batch_size, :], labels[j:j+micro_train_batch_size, :], return_token_entropy=False, attn_mask=attention_mask[j:j+micro_train_batch_size, :])["log_probs"]
                print("new_log_probs in", new_log_probs.device)
                print("new_log_probs shape", new_log_probs.shape)
                print("-"*100)
                print("after get_response_log_probs:")
                print("input_ids in", input_ids.device)
                print("attention_mask in", attention_mask[j:j+micro_train_batch_size, :].device)
                print("input_ids shape", input_ids[j:j+micro_train_batch_size, :].shape)
                print("attention_mask shape", attention_mask[j:j+micro_train_batch_size, :].shape)
                print("labels in", labels[j:j+micro_train_batch_size, :].device)
                print("labels shape", labels[j:j+micro_train_batch_size, :].shape)

                print("-"*100)
                exit()
                



        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        exit()

            

        


if __name__ == "__main__":
    main()


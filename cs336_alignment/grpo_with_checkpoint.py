import os
# # 使用v1版本, 拿到原始softmax的分布而不是topk等调整后的分布
# # 注意，这里仍存在问题，因为rollout的分布是topk/temperature调整后的分布，而不是原始softmax的分布
# # 这可能会导致轻微的“off policy”问题，但目前看来影响不大
# os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
## 关掉避免illegal memory access
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_DISABLE_FLASHINFER"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
from grpo_utility import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from drgrpo_grader import r1_zero_reward_fn
import gc
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
import time
import grpo_evaluate as grpo_evaluate
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import wandb
import bitsandbytes as bnb
from itertools import islice





LOCAL_SNAPSHOT = snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_files_only=True,      
    revision="main"             
)

n_grpo_steps = 300
learning_rate = 1e-5
warmup_steps = 10
lr_min = 5e-6
advantage_eps = 1e-6
sampling_min_tokens = 0
sampling_max_tokens = 512
epoch_per_rollout_batch = 3
train_batch_size = 64
rollout_batch_size = 64
gradient_accumulation_steps = 32
gpu_memory_utilization = 0.80
loss_type = "grpo_clip"
group_size = 8

## 目前的逻辑是这样的，仅为暂时的
num_optimizer_steps = n_grpo_steps * epoch_per_rollout_batch

model_version = "v7_L4"


use_std_normalization = False

grpo_clip_range = 0.30

### 断点需要注意,id, 模型，优化器，lr scheduler等都要一致, 要设置start_train_step
## 记得修改id
start_train_step = 1
Load_From_Checkpoint = False
checkpoint_dir = "grpo_checkpoint/grpo_clip/v4_L4_tried4_60"
if Load_From_Checkpoint:
    wandb.init(
    project="grpo",
    name=f"grpo_{loss_type}_{model_version}",
    id="rm4mfk3m",
    resume="allow",
    config={
        "loss_type": loss_type,
        "n_grpo_steps": n_grpo_steps,
        "epoch_per_rollout_batch": epoch_per_rollout_batch,
        "train_batch_size": train_batch_size,
        "rollout_batch_size": rollout_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "group_size": group_size,
        "use_std_normalization": use_std_normalization,
        "learning_rate": learning_rate,
        "grpo_clip_range": grpo_clip_range
    }, 
    tags=["fixed_mask_mean"])
else:
    wandb.init(
    project="grpo",
    name=f"grpo_{loss_type}_{model_version}",
    config={
        "loss_type": loss_type,
        "n_grpo_steps": n_grpo_steps,
        "epoch_per_rollout_batch": epoch_per_rollout_batch,
        "train_batch_size": train_batch_size,
        "rollout_batch_size": rollout_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "group_size": group_size,
        "use_std_normalization": use_std_normalization,
        "learning_rate": learning_rate,
        "grpo_clip_range": grpo_clip_range
    }, 
    tags=["fixed_mask_mean"])
wandb.log({}, commit=True)




def main():
    if Load_From_Checkpoint:
        resume_state = load_trainer_state(checkpoint_dir)
        rng = resume_state.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        random.seed(88)
        np.random.seed(88)
        torch.manual_seed(88)
        torch.cuda.manual_seed_all(88)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ## 64 % 64 == 0
    # assert train_batch_size % gradient_accumulation_steps == 0, (
    #     "train_batch_size must be divisible by gradient_accumulation_steps"
    # )
    ## 64/ 64 = 1
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    ## 64 % 8 == 0
    # assert rollout_batch_size % group_size == 0, (
    #     "rollout_batch_size must be divisible by group_size"
    # )
    ## 64/ 8 = 8
    n_prompt_per_rollout_batch = rollout_batch_size // group_size

    ## 64 >= 8
    # assert train_batch_size >= group_size, (
    #     "train_batch_size must be greater than or equal to group_size"
    # )

    ## 64 / 1 == 64
    n_micro_batches_per_epoch = rollout_batch_size // micro_train_batch_size

    wandb.define_metric("train_step")
    wandb.define_metric("raw_reward/*", step_metric="train_step")
    wandb.define_metric("advantage/*", step_metric="train_step")
    wandb.define_metric("eval_metrics/*", step_metric="train_step")
    wandb.define_metric("total_loss", step_metric="train_step")
    wandb.define_metric("token/*", step_metric="train_step")

    wandb.define_metric("optimizer_step")
    wandb.define_metric("train/*", step_metric="optimizer_step")


    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,
        seed=99,
        # logprobs=1
    )

    prompts, answers_pure = get_gsm8k_train_ready_prompts()
    dataset = list(zip(prompts, answers_pure))
    ## 8 prompts per batch


    dataloader = build_dataloader(dataset, n_prompt_per_rollout_batch)
    dataloader_iter = iter(dataloader)
    skip = (start_train_step-1) % len(dataloader)
    dataloader_iter = islice(dataloader_iter, skip, None)
    # batch = next(dataloader_iter)




    optimizer_step = 0
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_SNAPSHOT)
    tokenizer.padding_side = "left"
    for train_step in range(start_train_step, n_grpo_steps):
        print("-"*100)
        print(f"grpo step {train_step} starts:")
        batch = next(dataloader_iter)
        prompts, answers_pure = batch

        if train_step == 1:
            policy = AutoModelForCausalLM.from_pretrained(
                LOCAL_SNAPSHOT,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2"
            )
            optimizer = bnb.optim.PagedAdamW8bit(
                params=policy.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.0)
            scheduler = build_scheduler(optimizer, warmup_steps, n_grpo_steps*epoch_per_rollout_batch, lr_min)
            llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
            wandb.watch(policy, log="gradients", log_freq=40)
        elif Load_From_Checkpoint and train_step == start_train_step and train_step != 1:
            policy = AutoModelForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True, torch_dtype="bfloat16", attn_implementation="flash_attention_2")
            optimizer = bnb.optim.PagedAdamW8bit(
                params=policy.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.0)
            scheduler = build_scheduler(optimizer, warmup_steps, n_grpo_steps*epoch_per_rollout_batch, lr_min)
            # resume_state = load_trainer_state(checkpoint_dir)
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            optimizer_step = resume_state["optimizer_step"]

            
            print(f"[RESUME] from {checkpoint_dir}, start_train_step={start_train_step}, optimizer_step={optimizer_step}")

            llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
            wandb.watch(policy, log="gradients", log_freq=40)
        else:
            ## policy still on cpu
            llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)


        texts, token_ids = rollout_vllm(llm, prompts, sampling_params)
        # expanded_old_log_probs = []
        # for step_logprobs, step_token_ids in zip(old_log_probs, token_ids):
        #     seq_log_probs = []
        #     for per_step_dict, tid in zip(step_logprobs, step_token_ids):
        #             seq_log_probs.append(per_step_dict[tid].logprob)
        #     expanded_old_log_probs.append(seq_log_probs)

        len_token_ids = [len(x) for x in token_ids]
        max_token_len = max(len_token_ids)
        mean_token_len = sum(len_token_ids) / (len(len_token_ids)+ 1e-6)
        min_token_len = min(len_token_ids)




        answers_repeated = [x for x in answers_pure for _ in range(group_size)]
        prompt_repeated = [x for x in prompts for _ in range(group_size)]
        advantage, raw_reward, _ = compute_group_normalized_rewards(r1_zero_reward_fn, texts, answers_repeated, group_size, advantage_eps, use_std_normalization)
        wandb.log({
            "train_step": train_step,
            "raw_reward/mean": raw_reward.mean().item(),
            "advantage/mean": advantage.mean().item(),
            "advantage/max": advantage.max().item(),
            "token/max": max_token_len,
            "token/mean": mean_token_len,
            "token/min": min_token_len

        }, commit=False)
        prompt_and_response = [x + y for x, y in zip(prompt_repeated, texts)]
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

        labels = input_ids[:, 1:].to("cuda")



        policy.to("cuda")
        
        gradient_accumulation_count = 0

        old_log_probs_list = []

        for i in range(epoch_per_rollout_batch):
            non_zero_loss_count = 0
            mean_token_entropy = 0.0
            total_loss = 0.0
            for j in range(0, rollout_batch_size, micro_train_batch_size):
                # Count, even the skip case.
                gradient_accumulation_count += 1
                advantages_this_batch = advantage[j:j+micro_train_batch_size].to("cuda")
                if advantages_this_batch.abs().max() < 0.0001:
                    # print(f"skip in grpo step {train_step} batch {i} micro batch {j}")
                    old_log_probs_list.append(None)
                    if gradient_accumulation_count == gradient_accumulation_steps:
                        optimizer_step += 1
                        print(f"optimizer step {optimizer_step}")
                        if non_zero_loss_count != 0:
                            scheduler.step()
                        optimizer.step()

                        wandb.log({
                            "optimizer_step": optimizer_step,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/non_zero_loss_count": non_zero_loss_count,
                            "train/loss_mean": (total_loss / non_zero_loss_count) if non_zero_loss_count != 0 else 0,
                            "train/mean_token_entropy": mean_token_entropy / non_zero_loss_count if non_zero_loss_count != 0 else 0
                        }, commit=True)
                        optimizer.zero_grad(set_to_none=True)
                        gradient_accumulation_count = 0
                    continue
                non_zero_loss_count += 1
                advantages_this_batch = advantages_this_batch.unsqueeze(-1)


                out = get_response_log_probs(policy, input_ids[j:j+micro_train_batch_size, :], labels[j:j+micro_train_batch_size, :], return_token_entropy=True, attn_mask=attention_mask[j:j+micro_train_batch_size, :])
                new_log_probs = out["log_probs"]
                entropy = out["token_entropy"]
                # old_log_probs_this_batch = expanded_old_log_probs[j:j+micro_train_batch_size]


                response_lengths = len_token_ids[j:j+micro_train_batch_size]



                response_mask, new_lpg_probs_resp, masked_token_entropy = get_response_log_probs_tensor_and_response_mask(
                    new_log_probs,
                    response_lengths,
                    entropy
                )
                mean_token_entropy += masked_token_entropy.item()

                if loss_type == "grpo_clip" and i == 0:
                    old_log_probs_list.append(new_lpg_probs_resp.detach().clone())
                elif loss_type == "grpo_clip" and i != 0:
                    old_log_probs_tensor = old_log_probs_list[j//micro_train_batch_size]


                if loss_type == "no_baseline":
                    loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, loss_type, raw_rewards=raw_reward[j:j+micro_train_batch_size].unsqueeze(-1).to("cuda"))
                elif loss_type == "reinforce_with_baseline":
                    loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, loss_type, advantages=advantages_this_batch)
                elif loss_type == "grpo_clip":
                    if i == 0:
                        ## 第一轮，理论上退化成REINFORCE with baseline.
                        loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, "reinforce_with_baseline", advantages=advantages_this_batch)
                    else:
                        loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, loss_type, advantages=advantages_this_batch, old_log_probs=old_log_probs_tensor, cliprange=grpo_clip_range)

                # print(f"loss in grpo step {train_step} batch {i} micro batch {j}", loss)
                # print("-"*100)
                total_loss += loss.detach().mean().item()
                if gradient_accumulation_count == gradient_accumulation_steps:

                    optimizer_step += 1
                    print(f"optimizer step {optimizer_step}")
                    optimizer.step()
                    scheduler.step()
                    wandb.log({
                        "optimizer_step": optimizer_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/non_zero_loss_count": non_zero_loss_count,
                        "train/loss_mean": (total_loss / non_zero_loss_count) if non_zero_loss_count != 0 else 0,
                        "train/mean_token_entropy": mean_token_entropy / non_zero_loss_count if non_zero_loss_count != 0 else 0
                    }, commit=True)
                    optimizer.zero_grad(set_to_none=True)
                    gradient_accumulation_count = 0
            print(f"non_zero_loss_count in grpo step {train_step}, epoch {i}:", non_zero_loss_count)
        print(f"total_loss on train step {train_step}: {total_loss}")



        if (train_step) % 30 == 0:
            policy.save_pretrained(f"grpo_checkpoint/{loss_type}/{model_version}_{train_step}")
            save_checkpoint(f"grpo_checkpoint/{loss_type}/{model_version}_{train_step}", policy, optimizer, scheduler, train_step, optimizer_step)
            print("model saved in grpo step:", train_step)
        if (train_step) % 3 == 0:
            print("evaluating model in grpo step:", train_step)
            policy.to("cpu")
            del input_ids, attention_mask, labels
            del prompt_and_response_token_ids
            del advantage, raw_reward
            del texts, answers_repeated, prompt_repeated, prompt_and_response
            # del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
            correct_count, accuracy = grpo_evaluate.evaluate(llm, f"grpo_{loss_type}_{model_version}_{train_step}", r1_zero_reward_fn, max_tokens=sampling_max_tokens)
            wandb.log({
                "train_step": train_step,
                "eval_metrics/accuracy": accuracy,
                "eval_metrics/correct_count": correct_count,
                "total_loss": total_loss
            }, commit=True)
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            policy.to("cpu")
        ## free gpu memory here
            del input_ids, attention_mask, labels
            del prompt_and_response_token_ids
            del advantage, raw_reward
            del texts, answers_repeated, prompt_repeated, prompt_and_response
            # del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            wandb.log({"train_step": train_step}, commit=True)
                
            


if __name__ == "__main__":
    main()
    wandb.finish()

import os
# # 使用v1版本, 拿到原始softmax的分布而不是topk等调整后的分布
# # 注意，这里仍存在问题，因为rollout的分布是topk/temperature调整后的分布，而不是原始softmax的分布
# # 这可能会导致轻微的“off policy”问题，但目前看来影响不大
# os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from grpo_utility import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
# from sft_evaluate import init_vllm, load_policy_into_vllm_instance, evaluate_vllm
from torch.utils.data import DataLoader
from drgrpo_grader import r1_zero_reward_fn
import gc
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
import time
import grpo_evaluate as grpo_evaluate
from torch.nn.utils.rnn import pad_sequence

import wandb
# import bitsandbytes as bnb

#下载本地参数
# huggingface-cli download Qwen/Qwen2.5-Math-1.5B --revision main



LOCAL_SNAPSHOT = snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_files_only=True,      
    revision="main"             
)

n_grpo_steps = 200
learning_rate = 1e-5
advantage_eps = 1e-8
sampling_min_tokens = 4
sampling_max_tokens = 768
epoch_per_rollout_batch = 1
train_batch_size = 512
rollout_batch_size = 512
gradient_accumulation_steps = 64
gpu_memory_utilization = 0.80
loss_type = "reinforce_with_baseline"
group_size = 16

## 目前的逻辑是这样的，仅为暂时的
num_optimizer_steps = n_grpo_steps * epoch_per_rollout_batch

model_version = "v2"


use_std_normalization = True

grpo_clip_range = 0.05

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
    })

def main():



    ## 64 % 64 == 0
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    ## 512/ 128 = 4
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    ## 512 % 16 == 0
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    ## 512/ 16 = 32
    n_prompt_per_rollout_batch = rollout_batch_size // group_size

    ## 512 >= 16
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    ## 512 / 4 == 128
    # n_micro_batches_per_epoch = rollout_batch_size // micro_train_batch_size

    wandb.define_metric("train_step")
    wandb.define_metric("raw_reward/*", step_metric="train_step")
    wandb.define_metric("advantage/*", step_metric="train_step")
    wandb.define_metric("eval_metrics/*", step_metric="train_step")
    wandb.define_metric("total_loss", step_metric="train_step")

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
        logprobs=1
    )

    prompts, answers_pure = get_gsm8k_train_ready_prompts()
    dataset = list(zip(prompts, answers_pure))
    ## 8 prompts per batch
    dataloader = DataLoader(
        dataset,
        batch_size=n_prompt_per_rollout_batch,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True
    )

    dataloader_iter = iter(dataloader)

    optimizer_step = 0
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_SNAPSHOT)
    tokenizer.padding_side = "left"
    for train_step in range(n_grpo_steps):
        print("-"*100)
        print(f"grpo step {train_step} starts:")
        batch = next(dataloader_iter)
        prompts, answers_pure = batch
        if train_step == 0:
            policy = AutoModelForCausalLM.from_pretrained(
                LOCAL_SNAPSHOT,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2"
            ).to("cuda")
            optimizer = torch.optim.AdamW(
                policy.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                fused=True
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=50,
                num_training_steps=num_optimizer_steps
            )
            llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
            wandb.watch(policy, log="gradients", log_freq=40)
        elif train_step % 10 == 0:
            ## vllm loaded when evaluating, do not need to load again
            pass
        else:
            ## policy still on cuda for h200
            ## keep llm
            # llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)


        texts, old_log_probs, token_ids = rollout_vllm(llm, prompts, sampling_params)
        expanded_old_log_probs = []
        for step_logprobs, step_token_ids in zip(old_log_probs, token_ids):
            seq_log_probs = []
            for per_step_dict, tid in zip(step_logprobs, step_token_ids):
                    seq_log_probs.append(per_step_dict[tid].logprob)
            expanded_old_log_probs.append(seq_log_probs)


        answers_repeated = [x for x in answers_pure for _ in range(group_size)]
        prompt_repeated = [x for x in prompts for _ in range(group_size)]
        advantage, raw_reward, _ = compute_group_normalized_rewards(r1_zero_reward_fn, texts, answers_repeated, group_size, advantage_eps, use_std_normalization)
        wandb.log({
            "train_step": train_step,
            "raw_reward/mean": raw_reward.mean().item(),
            "advantage/mean": advantage.mean().item(),
            "advantage/max": advantage.max().item()
        }, commit=False)
        prompt_and_response = [x + y for x, y in zip(prompt_repeated, texts)]
        prompt_and_response_token_ids = tokenizer(
            prompt_and_response,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # del llm
        # gc.collect()
        # torch.cuda.empty_cache()

        input_ids = prompt_and_response_token_ids["input_ids"].to("cuda")
        attention_mask = prompt_and_response_token_ids["attention_mask"].to("cuda")

        labels = input_ids[:, 1:].to("cuda")



        # policy.to("cuda")
        
        gradient_accumulation_count = 0
        # optimizer = torch.optim.Adam(policy.parameters(),
        #     lr=learning_rate,
        #     betas=(0.9, 0.95),
        #     weight_decay=0.0)
        total_loss = 0.0
        for i in range(epoch_per_rollout_batch):
            non_zero_loss_count = 0
            for j in range(0, rollout_batch_size, micro_train_batch_size):
                # Count, even the skip case.
                gradient_accumulation_count += 1
                advantages_this_batch = advantage[j:j+micro_train_batch_size].to("cuda")
                if advantages_this_batch.abs().max() < 0.0001:
                    print(f"skip in grpo step {train_step} batch {i} micro batch {j}")
                    if gradient_accumulation_count == gradient_accumulation_steps:
                        optimizer_step += 1
                        print(f"optimizer step {optimizer_step}")
                        optimizer.step()
                        scheduler.step()
                        wandb.log({
                            "optimizer_step": optimizer_step,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/non_zero_loss_count": non_zero_loss_count,
                            "train/loss_mean": (total_loss / non_zero_loss_count) if non_zero_loss_count != 0 else 0
                        }, step=optimizer_step, commit=True)
                        optimizer.zero_grad(set_to_none=True)
                        gradient_accumulation_count = 0
                    continue
                non_zero_loss_count += 1
                advantages_this_batch = advantages_this_batch.unsqueeze(-1)


                new_log_probs = get_response_log_probs(policy, input_ids[j:j+micro_train_batch_size, :], labels[j:j+micro_train_batch_size, :], return_token_entropy=False, attn_mask=attention_mask[j:j+micro_train_batch_size, :])["log_probs"]
                old_log_probs_this_batch = expanded_old_log_probs[j:j+micro_train_batch_size]


                old_log_probs_tensor, response_mask, new_lpg_probs_resp = get_response_log_probs_tensor_and_response_mask(new_log_probs, old_log_probs_this_batch)


                if loss_type == "no_baseline":
                    loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, loss_type, raw_rewards=raw_reward[j:j+micro_train_batch_size].unsqueeze(-1).to("cuda"))
                elif loss_type == "reinforce_with_baseline":
                    loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, loss_type, advantages=advantages_this_batch)
                elif loss_type == "grpo_clip":
                    loss, _ = grpo_microbatch_train_step(new_lpg_probs_resp, response_mask, gradient_accumulation_steps, loss_type, advantages=advantages_this_batch, old_log_probs=old_log_probs_tensor, cliprange=grpo_clip_range)
                print(f"loss in grpo step {train_step} batch {i} micro batch {j}", loss)
                print("-"*100)
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
                        "train/loss_mean": (total_loss / non_zero_loss_count) if non_zero_loss_count != 0 else 0
                    }, step=optimizer_step, commit=True)
                    optimizer.zero_grad(set_to_none=True)
                    gradient_accumulation_count = 0
            print(f"non_zero_loss_count in grpo step {train_step}, epoch {i}:", non_zero_loss_count)
        print(f"total_loss on train step {train_step}: {total_loss}")





        if (train_step+1) % 50 == 0:
            policy.save_pretrained(f"grpo_checkpoint/{loss_type}/{model_version}_{train_step+1}")
            print("model saved in grpo step:", train_step)
        if (train_step+1) % 10 == 0:
            print("evaluating model in grpo step:", train_step)
            # policy.to("cpu")
            # del input_ids, attention_mask, labels
            # del prompt_and_response_token_ids
            # del advantage, raw_reward, expanded_old_log_probs
            # del texts, answers_repeated, prompt_repeated, prompt_and_response
            # # del optimizer
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # llm = init_vllm(LOCAL_SNAPSHOT, "cuda", 99, gpu_memory_utilization=gpu_memory_utilization)
            load_policy_into_vllm_instance(policy, llm)
            correct_count, accuracy = grpo_evaluate.evaluate(llm, f"grpo_{loss_type}_{model_version}_{train_step+1}", r1_zero_reward_fn, max_tokens=sampling_max_tokens)
            wandb.log({
                "train_step": train_step,
                "eval_metrics/accuracy": accuracy,
                "eval_metrics/correct_count": correct_count,
                "total_loss": total_loss
            }, commit=False)
            # del llm
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
        else:
            # policy.to("cpu")
        ## free gpu memory here
            # del input_ids, attention_mask, labels
            # del prompt_and_response_token_ids
            # del advantage, raw_reward, expanded_old_log_probs
            # del texts, answers_repeated, prompt_repeated, prompt_and_response
            # del optimizer
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            pass
                
            

if __name__ == "__main__":
    main()
    wandb.finish()


from grpo_utility import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from sft_evaluate import init_vllm, load_policy_into_vllm_instance, evaluate_vllm
from grpo_utility import get_gsm8k_train_ready_prompts
from torch.utils.data import DataLoader
from drgrpo_grader import r1_zero_reward_fn


n_grpo_steps = 200
learning_rate = 1e-5
advantage_eps = 1e-8
sampling_min_tokens = 4
sampling_max_tokens = 1024
epoch_per_rollout_batch = 1
train_batch_size = 128
rollout_batch_size = train_batch_size
gradient_accumulation_steps = 32
gpu_memory_utilization = 0.85
loss_type = "reinforce_with_baseline"
group_size = 4

use_std_normalization = True

optimizer = torch.optim.Adam(policy.parameters(),
                            lr=learning_rate,
                            betas=(0.9, 0.95,
                            weight_decay=0.0))



assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
)
## 128/ 32 = 4
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
)
## 128/ 4 = 32
n_prompt_per_rollout_batch = train_batch_size // 4
assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
)

n_micro_batches_per_epoch = rollout_batch_size // micro_train_batch_size

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=sampling_max_tokens,
    min_tokens=sampling_min_tokens,
    stop=["</answer>"],
    include_stop_str_in_output=True,
    n=group_size,
    seed=99
)

i, j = 0, 0
prompts, answers_pure = get_gsm8k_train_ready_prompts()
dataset = list(zip(prompts, answers_pure))
## 32 prompts per batch
dataloader = DataLoader(
    dataset,
    batch_size=n_prompt_per_rollout_batch,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

dataloader_iter = iter(dataloader)


for train_step in range(n_grpo_steps//n_micro_batches_per_epoch):
    batch = next(dataloader_iter)
    prompts, answers_pure = batch
    if train_step == 0:
        policy = AutoModelForCausalLM.from_pretrained(
            f"Model/policy_{train_step}",
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
    
    ## 现在rollout, 32 prompts得到32*4=128个response
    raw_responses = llm.generate(prompts, sampling_params)
    outputs = raw_responses.outputs
    answers_repeated = [x for x in answers_pure for _ in range(group_size)]
    # reponse_list_by_group = [raw_responses[i:i+group_size] for i in range(0, len(raw_responses), group_size)]
    ## 32
    for i in range(gradient_accumulation_steps):
        advantages, _, _ = compute_group_normalized_rewards(r1_zero_reward_fn, outputs, answers_repeated, group_size, advantage_eps, use_std_normalization)
        
        
        

    






policy = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2"
)




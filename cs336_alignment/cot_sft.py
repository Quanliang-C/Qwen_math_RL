from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import re
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
import random
import math
from string import Template

random.seed(99)

system_prompt = Template(Path("cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8"))

ds = load_from_disk("data/cleaning_cot_sft")
# ds = ds.filter(lambda x: x["conversations"][-1]["value"].count("</answer>") == 1)
ds = ds.map(lambda x: {"conversations": x["conversations"][-1]["value"]})

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B",
                                            torch_dtype="bfloat16",
                                            attn_implementation="flash_attention_2").to("cuda")


model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()



prompt_strs = [system_prompt.substitute(question=x["problem"]) for x in ds]
output_strs = [x["conversations"] for x in ds]


def tokenize_prompt_and_output(prompts_str: list[str], outputs_str: list[str], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    # 这里没有使用return pt, 这里的是list
    prompts_token_ids = tokenizer(prompts_str, padding=False)["input_ids"]
    outputs_token_ids = tokenizer(outputs_str, padding=False, add_special_tokens=False)["input_ids"]
    
    seq_len_prompts = [len(x) for x in prompts_token_ids]

    seq_all = [torch.tensor(prompt + output).long() for prompt, output in zip(prompts_token_ids, outputs_token_ids)]
    pad_id = tokenizer.pad_token_id

    seq_all = pad_sequence(seq_all, batch_first=True, padding_value=pad_id)
    attn_mask = (seq_all != pad_id)

    input_ids = seq_all[:, :-1].contiguous()
    labels = seq_all[:, 1:].contiguous()
    attn_for_labels = attn_mask[:, 1:]

    B, L = labels.size()
    prompt_len = torch.tensor(seq_len_prompts, device=labels.device, dtype=torch.long).unsqueeze(1)
    cols = torch.arange(L, device=labels.device).unsqueeze(0).expand(B, L)
    response_mask = (cols >= (prompt_len - 1))

    response_mask = response_mask & attn_for_labels
    response_mask = response_mask.long()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def collate_fn(batch):
    # ### 修改3: 新增collate函数（运行在DataLoader worker里）
    # batch: List[Tuple[prompt_str, output_str]]
    prompts, outputs = zip(*batch)
    p_tok = tokenizer(list(prompts), add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    o_tok = tokenizer(list(outputs), add_special_tokens=False, padding=False, truncation=False)["input_ids"]

    pad_id = tokenizer.pad_token_id
    inps, lbls, rmask = [], [], []
    for p, o in zip(p_tok, o_tok):
        seq = torch.tensor(p + o, dtype=torch.long)
        inp = seq[:-1]
        lbl = seq[1:]
        L = lbl.size(0)
        prompt_len = len(p)
        resp_mask = torch.arange(L).ge(prompt_len - 1).to(torch.long)  # 只评估回答段
        inps.append(inp)
        lbls.append(lbl)
        rmask.append(resp_mask)

    input_ids = pad_sequence(inps, batch_first=True, padding_value=pad_id)
    labels    = pad_sequence(lbls, batch_first=True, padding_value=pad_id)  # ### 修改4: 这里用pad_id占位；后续用response_mask屏蔽掉
    response_mask = pad_sequence(rmask, batch_first=True, padding_value=0)
    attention_mask = (input_ids != pad_id)
    return input_ids, labels, response_mask, attention_mask

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    return -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


## 在这里增加传入attn_mask
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
    logits = model(input_ids, attention_mask=attn_mask).logits
    log_prob = torch.log_softmax(logits, dim=-1)
    log_prob = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        return {"log_probs": log_prob, "token_entropy": token_entropy}
    else:
        return {"log_probs": log_prob}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    mask_sum = mask.to(tensor.dtype)
    masked_tensor = tensor * mask.to(tensor.dtype)
    if dim is None:
        mask_sum = mask_sum.sum()
        s = masked_tensor.sum()/mask_sum
    else:
        mask_sum = mask_sum.sum(dim=dim)
        s = masked_tensor.sum(dim=dim)/mask_sum

    return s / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size = policy_log_probs.shape[0]
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant=float(normalize_constant), dim=-1)
    loss = loss.mean()
    loss = -(loss/float(gradient_accumulation_steps))
    loss.backward()
    return loss, {}



def log_generations(loss: torch.Tensor, epoch: int, step: int):
    print(f"Epoch {epoch}, step {step}, training loss:", loss.item())



# input_ids, labels, response_mask = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer).values()

# dataset = list(zip(input_ids, labels, response_mask))
# ### 修改5: 数据集改为字符串pair，分词挪到collate中做
dataset = list(zip(prompt_strs, output_strs))
# subset = Subset(dataset, list(range(256)))
random.seed(99)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
batch_size = 2
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,              # ### 修改6: 使用collate_fn做在线分词与动态padding
    num_workers=4,                      # ### 修改7: 开worker并行分词/I-O
    pin_memory=True,                    # ### 修改8: 固定页内存，配合non_blocking异步H2D
    persistent_workers=True,            # ### 修改9: 复用worker，减少每个epoch的热身开销
    prefetch_factor=4,                  # ### 修改10: 每个worker预取多个batch
)

gradient_accumulation_steps = 2
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 5

steps_per_epoch = math.ceil(len(dataset) / batch_size)
total_steps = epochs * steps_per_epoch // gradient_accumulation_steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim,
    T_max=total_steps,
    eta_min=1e-6)

optim_step = 0
step = 0
for epoch in range(epochs):
    batch_loss = 0
    for i, batch in enumerate(dataloader):
        step += 1
        input_ids, labels, response_mask, attention_mask = batch
        input_ids, labels, response_mask, attention_mask = input_ids.to("cuda", non_blocking=True), labels.to("cuda", non_blocking=True), response_mask.to("cuda", non_blocking=True), attention_mask.to("cuda", non_blocking=True)
        ret = get_response_log_probs(model, input_ids, labels, return_token_entropy=True, attn_mask=attention_mask)
        response_log_probs = ret["log_probs"]
        token_entropy = ret["token_entropy"]
        loss, info = sft_microbatch_train_step(response_log_probs, response_mask, gradient_accumulation_steps=gradient_accumulation_steps, normalize_constant=1.0)
        batch_loss += loss.item()
        print("step", step, "token_entropy", token_entropy.mean().item())
        if (i+1) % gradient_accumulation_steps == 0:
            optim.step()
            optim_step += 1
            optim.zero_grad()
            scheduler.step()
            log_generations(loss, epoch, optim_step)
    print(f"Epoch {epoch}, mean training loss:", batch_loss/len(dataloader))


model.config.use_cache = True
model.save_pretrained("models/cot_sft_v2")



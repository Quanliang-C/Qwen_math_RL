from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import re
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
import random
import math

random.seed(99)

system_prompt = Path("cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8")

ds = load_from_disk("data/cleaning_cot_sft")
# ds = ds.filter(lambda x: x["conversations"][-1]["value"].count("</answer>") == 1)
ds = ds.map(lambda x: {"conversations": x["conversations"][-1]["value"]})

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B",
                                            torch_dtype="bfloat16",
                                            attn_implementation="flash_attention_2").to("cuda")



prompt_strs = [system_prompt.format(question=x["problem"]) for x in ds]
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



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    return -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
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
    logits = model(input_ids).logits
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
    masked_tensor = tensor * mask.to(tensor.dtype)
    s = masked_tensor.sum() if dim is None else masked_tensor.sum(dim=dim)
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
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant=float(normalize_constant))
    loss = -(loss/float(gradient_accumulation_steps))/float(batch_size)
    loss.backward()
    return loss, {}



def log_generations(loss: torch.Tensor, epoch: int):
    print(f"Epoch {epoch}, training loss:", loss.item())



input_ids, labels, response_mask = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer).values()

dataset = list(zip(input_ids, labels, response_mask))
# subset = Subset(dataset, list(range(256)))
random.seed(99)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



gradient_accumulation_steps = 2
optim = torch.optim.Adam(model.parameters(), lr=5e-6)
epochs = 3

steps_per_epoch = math.ceil(len(dataset) / 1)
total_steps = epochs * steps_per_epoch // gradient_accumulation_steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim,
    T_max=total_steps,
    eta_min=1e-7)

for epoch in range(epochs):
    for i, batch in enumerate(dataloader):
        input_ids, labels, response_mask = batch
        input_ids, labels, response_mask = input_ids.to("cuda"), labels.to("cuda"), response_mask.to("cuda")
        response_log_probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)["log_probs"]
        loss, info = sft_microbatch_train_step(response_log_probs, response_mask, gradient_accumulation_steps=4, normalize_constant=1.0)
        if (i+1) % gradient_accumulation_steps == 0:
            optim.step()
            optim.zero_grad()
            scheduler.step()
            log_generations(loss, epoch)

model.save_pretrained("models/cot_sft_v1")



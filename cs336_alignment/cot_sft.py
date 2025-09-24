from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import re

ds = load_from_disk("data/cleaning_cot_sft")
ds = ds.filter(lambda x: x["conversations"][-1]["value"].count("</answer>") == 1)
ds = ds.map(lambda x: {"conversations": x["conversations"][-1]["value"]})

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B",
                                            torch_dtype="bfloat16",
                                            attn_implementation="flash_attention_2")



system_prompt = Path("cs336_alignment/prompts/r1_zero.prompt").read_text(encoding="utf-8")
prompts_str = [system_prompt.format(question=x["problem"]) for x in ds]

outputs_str = [x["conversations"][-1]["value"] for x in ds]



def tokenize_prompt_and_output(prompts_str: list[str], outputs_str: list[str], tokenizer: AutoTokenizer) -> dict[str, torch.Tensor]:
    





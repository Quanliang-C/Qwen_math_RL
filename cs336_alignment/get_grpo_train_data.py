import random

random.seed(999)

with open("data/gsm8k/train.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)
lines = lines[:100]

with open("data/gsm8k/train_grpo_L4.jsonl", "w", encoding="utf-8") as w:
    w.writelines(lines)

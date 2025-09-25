import numpy as np
import json

with open("outputs/sft_metrics.jsonl", "r") as f:
    all_metrics = [json.loads(line) for line in f]


count_format_right = 0
count_right = 0

for metric in all_metrics:
    if metric["reward"] == 1.0:
        print(metric["index"])
        print(metric["response"])
        print(metric["ground_truth"])
        print("--------------------------------")
        count_right += 1
    elif metric["reward"] == 0.0 and metric["format_reward"] == 1.0:
        print("format right but answer wrong")
        print(metric["index"])
        print(metric["response"])
        print(metric["ground_truth"])
        print("--------------------------------")
        count_format_right += 1


print(f"count_right: {count_right}")
print(f"count_format_right: {count_format_right}")
print(f"count_right / (count_right + count_format_right): {count_right / (count_right + count_format_right)}")
from datasets import load_dataset
import json
import re
from transformers import AutoTokenizer
from typing import Callable


ds = load_dataset("open-r1/OpenThoughts-114k-math")
ds = ds["train"]


ds = ds.filter(lambda x: x["correct"] is True and int(x.get("generated_token_count", 0)) < 1024)

def replace_boxed(text: str) -> str:
    """
    将所有 \boxed{...} 替换为 <answer>...</answer>。
    规则极简：只做花括号计数匹配；不处理转义、不理解 LaTeX 语义。
    支持任意层 { } 嵌套。
    """
    i, n = 0, len(text)
    out = []
    token = r'\boxed'

    while i < n:
        j = text.find(token, i)
        if j == -1:
            out.append(text[i:])
            break

        # 先写入 token 之前的内容
        out.append(text[i:j])

        k = j + len(token)
        # 跳过空白
        while k < n and text[k].isspace():
            k += 1

        # 期望紧跟左花括号
        if k < n and text[k] == '{':
            # 从左花括号位置开始做简单计数匹配
            depth = 0
            p = k
            while p < n:
                c = text[p]
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        # p 为匹配到的右花括号
                        inner = text[k+1:p]
                        out.append(f'<answer>{inner}</answer>')
                        i = p + 1
                        break
                p += 1
            else:
                # 没有配对成功：把当前字符输出一个，继续扫，避免卡死
                out.append(text[j])
                i = j + 1
        else:
            # 不是规范的 \boxed{...}，按普通字符处理一个字节
            out.append(text[j])
            i = j + 1
    return ''.join(out)


def to_sft(example):
    if isinstance(example.get("conversations"), list):
        new_convs = []
        for m in example["conversations"]:
            value = m.get("value")
            value = value.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
            value = replace_boxed(value)
            m["value"] = value
            new_convs.append(m)
        example["conversations"] = new_convs
    return example


ds = ds.map(to_sft)
ds = ds.remove_columns(["source", "solution", "messages", "system"])



print(ds)
print(ds[0])


ds.save_to_disk("data/cleaning_cot_sft")


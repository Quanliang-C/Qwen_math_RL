from datasets import load_dataset
import json
import re
from transformers import AutoTokenizer
from typing import Callable


ds = load_dataset("open-r1/OpenThoughts-114k-math")
ds = ds["train"]


ds = ds.filter(lambda x: x["correct"] is True and int(x.get("generated_token_count", 0)) < 1024)
# 暂时包含了不正确的答案
# ds = ds.filter(lambda x: int(x.get("generated_token_count", 0)) < 1024)

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


def wrap_boxed_as_answer(text: str) -> str:
    """
    将所有 \boxed{...} 外层包裹为 <answer>\boxed{...}</answer>。
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
                        # p 为匹配到的右花括号（含）
                        whole = text[j:p+1]  # 包含 \boxed 与其完整花括号
                        out.append(f'<answer>{whole}</answer>')
                        i = p + 1
                        break
                p += 1
            else:
                # 没配对成功：按一个普通字符输出，避免卡死
                out.append(text[j])
                i = j + 1
        else:
            # 不是规范的 \boxed{...}，按普通字符处理一个字节
            out.append(text[j])
            i = j + 1

    return ''.join(out)


def move_think_before_answer(text: str) -> str:
    """
    将最后一个位于首个 <answer> 之前的 '</think>' 移动到该 <answer> 的正前方，
    确保出现精确的 '</think> <answer>'（中间恰好一个空格）。
    若文本已包含 '</think> <answer>'，或缺少 '</think>' / '<answer>'，则原样返回。
    """
    # 已满足格式，直接返回
    if "</think> <answer>" in text:
        return text

    first_ans = text.find("<answer>")
    if first_ans == -1:
        return text  # 没有 <answer>，不处理

    # 在首个 <answer> 之前，寻找“最后一个” </think>
    last_think_close = text.rfind("</think>", 0, first_ans)
    if last_think_close == -1:
        return text  # 没有 </think>，不处理

    # 1) 删除这个 </think>
    before = text[:last_think_close]
    after  = text[last_think_close + len("</think>"):]

    # 2) 在首个 <answer> 前插入精确的 '</think> '（一个空格后紧跟 <answer>）
    ans_pos = after.find("<answer>")
    if ans_pos == -1:
        # 理论上不会发生（因为我们在原文中找到了 <answer>），稳妥起见：
        return before + after

    # 去掉 <answer> 前的尾随空白，保证中间恰好一个空格
    left  = re.sub(r"[ \t\r\f\v]*$", "", after[:ans_pos])
    right = after[ans_pos + len("<answer>"):]  # 丢弃原始的 "<answer>"，待会儿用标准串替换

    return before + left + "</think> <answer>" + right


def strip_solution_markers(text: str) -> str:
    """
    移除字面量标记 '<|begin_of_solution|>' 与 '<|end_of_solution|>'。
    仅去标记本身，不更改其中内容；为避免多余空行，可选做一次轻量空行压缩（可按需注释）。
    """
    s = text.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
    # 轻量清理：把连续的空行压成一行（如不需要可注释掉）
    s = re.sub(r"[ \t]*\n[ \t]*\n+", "\n", s)
    return s


_BEGIN_T = re.compile(r"<\|\s*begin_of_thought\s*\|>", re.IGNORECASE)
_END_T   = re.compile(r"<\|\s*end_of_thought\s*\|>",   re.IGNORECASE)

def replace_thought_tags(text: str) -> str:
    """
    将 <|begin_of_thought|> → <think>
       <|end_of_thought|>   → </think>
    同时把常见误写 <\\think> 规范为 </think>。
    只做字面替换，不改动中间内容与换行。
    """
    s = _BEGIN_T.sub("<think>", text)
    s = _END_T.sub("</think>", s)
    s = s.replace("<\\think>", "</think>")  # 纠正常见误写
    return s

def truncate_after_last_answer(text: str) -> str:
    """把文本截断到最后一个 </answer>（包含它）为止；找不到就原样返回。"""
    i = text.rfind("</answer>")
    return text if i == -1 else text[: i + len("</answer>")]

def to_sft(example):
    if isinstance(example.get("conversations"), list):
        new_convs = []
        for m in example["conversations"]:
            value = m.get("value")
            value = value.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
            value = wrap_boxed_as_answer(value)
            value = strip_solution_markers(value)
            value = replace_thought_tags(value)
            value = move_think_before_answer(value)
            value = truncate_after_last_answer(value)
            m["value"] = value
            new_convs.append(m)
        example["conversations"] = new_convs
    return example


ds = ds.map(to_sft)
ds = ds.filter(lambda x: x["conversations"][-1]["value"].count("</answer>") == 1)
ds = ds.filter(lambda x: x["conversations"][-1]["value"].count("</think>") == 1)
ds = ds.remove_columns(["source", "solution", "messages", "system"])



print(ds)
print(ds[0])


ds.save_to_disk("data/cleaning_cot_sft")


"""
Generate SFT dataset for Qwen2.5-3B from Countdown-Task-GOLD verified subsets.

- Load verified_Qwen2.5-7B-Instruct (30.4k)
- Load verified_Qwen3-4B-Instruct-2507 (~27k)
- Sample + deduplicate
- Extract clean <answer>...</answer>
- Build SFT format using your existing countdown data format
- Save as parquet to local/hdfs
"""

import re
import os
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
from collections import Counter
import ipdb


# ================================================================
#  Helper: extract <answer> ... </answer>
# ================================================================
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PREFIX_PATTERN = re.compile(r'^\s*<think>\s*', re.DOTALL)

def extract_answer(text):
    """Extract final equation inside <answer>...</answer>"""
    if text is None:
        return None
    m = ANSWER_PATTERN.search(text)
    if not m:
        return None
    return m.group(1).strip()

def strip_leading_think(text: str) -> str:
    """
    去掉开头的一个 <think> ...（以及紧跟的换行/空白），
    保留后面的 </think> 和 <answer>。
    如果没有 <think> 前缀，则原样返回。
    """
    if text is None:
        return text
    return THINK_PREFIX_PATTERN.sub('', text, count=1)

# ================================================================
#  Build prefix (your existing code)
# ================================================================
def make_prefix(dp, template_type):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

def clean_answer_keep_left(expr: str) -> str:
    """
    处理带 <answer>...</answer> 的完整 SFT 文本：
    1. 只保留等号左侧的表达式
    2. 去掉 <answer> 内部的所有换行符
    3. 保留其他内容（尤其是 <think>）不变
    """
    m = re.search(r"<answer>(.*?)</answer>", expr, flags=re.S)
    if not m:
        return expr  # 没找到 <answer> 就原样返回

    answer_content = m.group(1).strip()

    # 1) 仅保留等号左边的表达式
    left_expr = answer_content.split("=", 1)[0].strip()

    # 2) 去掉所有换行符、空行、前后空格
    left_expr_no_newline = re.sub(r"\s+", " ", left_expr).strip()

    # 3) 构造新的 <answer> ... </answer>
    new_answer = f"<answer>{left_expr_no_newline}</answer>"

    # 4) 替换原 answer
    return expr[:m.start()] + new_answer + expr[m.end():]

def operator_aware_weighted_sampling(merged, max_sft_samples, ratio_add_sub=0.4, ratio_mul=0.3, ratio_div=0.3):
    print("Performing operator-aware weighted sampling...")
    total_target = max_sft_samples

    # 1. 分桶
    only_add_sub = []
    has_mul = []
    has_div = []

    for idx, ex in enumerate(merged):
        eq = ex.get("equation", "")
        if eq is None:
            continue
        clean_eq = eq.strip()

        has_mul_flag = "*" in clean_eq
        has_div_flag = "/" in clean_eq

        if not has_mul_flag and not has_div_flag:
            only_add_sub.append(idx)
        if has_mul_flag:
            has_mul.append(idx)
        if has_div_flag:
            has_div.append(idx)

    print(f"only_add_sub: {len(only_add_sub)}")
    print(f"has_mul:      {len(has_mul)}")
    print(f"has_div:      {len(has_div)}")

    # 2. 分配配额
    n_add_sub = int(total_target * ratio_add_sub)    
    n_mul     = int(total_target * ratio_mul)
    n_div     = int(total_target * ratio_div)

    # 3. 从每个桶采样（不足则取全部）
    import random
    random.seed(42)

    def safe_sample(lst, k):
        if len(lst) <= k:
            return lst  # 全拿
        return random.sample(lst, k)

    sampled_add_sub = safe_sample(only_add_sub, n_add_sub)
    sampled_mul     = safe_sample(has_mul,     n_mul)
    sampled_div     = safe_sample(has_div,     n_div)

    # 4. 合并 + 去重
    merged_indices = list(set(sampled_add_sub + sampled_mul + sampled_div))
    random.shuffle(merged_indices)

    # 如仍超过 max_sft_samples，则再裁一次（几乎不可能，但保持健壮性）
    if len(merged_indices) > total_target:
        merged_indices = merged_indices[:total_target]

    print(f"Final sampled size = {len(merged_indices)}")

    merged = merged.select(merged_indices)
    return merged

def operator_presence_stats(merged):
    op_sample_counter = Counter()
    total_samples = len(merged)

    for ex in merged:
        eq = ex.get("equation")
        if not eq:
            continue
        if "+" in eq:
            op_sample_counter["+"] += 1
        if "-" in eq:
            op_sample_counter["-"] += 1
        if "*" in eq:
            op_sample_counter["*"] += 1
        if "/" in eq:
            op_sample_counter["/"] += 1

    print("Operator presence stats (sample-level):")
    for op in ["+", "-", "*", "/"]:
        cnt = op_sample_counter[op]
        ratio = cnt / total_samples
        print(f"  {op}: {cnt}/{total_samples} ({ratio:.4f})")
    


# ================================================================
#    MAIN
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/users/gwang16/Jillian/TinyZero/data/countdown_sft")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", default="base")
    parser.add_argument("--max_sft_samples", type=int, default=20000)
    parser.add_argument("--test_size", type=int, default=1024)
    args = parser.parse_args()

    # -------------------------
    # Load datasets
    # -------------------------
    print("Loading verified subsets...")
    ds_q25_7b = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "verified_Qwen2.5-7B-Instruct", split="train")
    ds_q3_4b = load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "verified_Qwen3-4B-Instruct-2507", split="train")

    # -------------------------
    # Merge + Extract Answer
    # -------------------------
    print("Extracting answers...")
    def extract_map(example):
        message = example.get("messages", [])
        assistant_answer = message[-1].get("content", "")
        role = message[-1].get("role", "")
        if role == "assistant":
            answer = assistant_answer.strip()
            example["org_answer"] = answer
            answer = clean_answer_keep_left(answer)
            example["answer"] = answer
            example["equation"] = extract_answer(answer)
        else:
            answer = None
            example["org_answer"] = answer
            example["answer"] = answer
            example["equation"] = None
        return example


    ds_q25_7b = ds_q25_7b.map(extract_map)
    ds_q3_4b = ds_q3_4b.map(extract_map)
    # ipdb.set_trace()

    # Remove bad entries (no answer)
    ds_q25_7b = ds_q25_7b.filter(lambda x: x["answer"] is not None)
    ds_q3_4b = ds_q3_4b.filter(lambda x: x["answer"] is not None)
    # ipdb.set_trace()
    # -------------------------
    # Deduplicate by (target, nums)
    # -------------------------
    def key_fn(x):
        return (x["target"], tuple(x["nums"]))

    print("Deduplicating...")
    merged = concatenate_datasets([ds_q25_7b, ds_q3_4b])
    seen = set()
    keep_indices = []

    for idx, ex in enumerate(merged):
        k = key_fn(ex)
        if k not in seen:
            seen.add(k)
            keep_indices.append(idx)

    merged = merged.select(keep_indices)
    print(f"After dedup: {len(merged)} samples")

    # ===== 样本级别统计加减乘除比例（推荐） =====
    operator_presence_stats(merged)
    # ipdb.set_trace()

    # -------------------------
    # Operator-aware weighted sampling
    # -------------------------
    merged = operator_aware_weighted_sampling(merged, args.max_sft_samples)
    operator_presence_stats(merged)
    # ipdb.set_trace()

    # -------------------------
    # Build SFT format
    # -------------------------
    print("Building SFT dataset format...")

    def build_sft(example, idx):
        prefix = make_prefix(example, args.template_type)

        # assistant_output = example["answer"].strip()

        # Build assistant output with <think>...</think><answer>...</answer>
        # assistant_output = (
        #     prefix +
        #     "</think>\n" +    # close think
        #     f"<answer>{example['answer']}</answer>"
        # )
        if example["target"] == 47 and example["nums"] == [51, 48, 92, 8]:
            ipdb.set_trace()

        # 2) answer：去掉开头那一个 <think>，让模型从 <think> 后第一个 token 开始学习
        raw_answer = example["answer"]
        if raw_answer is None:
            cleaned_answer = None
        else:
            cleaned_answer = strip_leading_think(raw_answer.strip())


        dp = {
            "data_source": "countdown_sft",
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "math",
            "answer": cleaned_answer,   # SFT label
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target": example["target"],
                    "numbers": example["nums"]
                }
            },
            "extra_info": {
                "index": idx,
            }
        }
        # ipdb.set_trace()
        return dp

    sft_dataset = merged.map(build_sft, with_indices=True)

    # -------------------------
    # Train/Test split
    # -------------------------
    TRAIN_SIZE = len(sft_dataset) - args.test_size
    train_dataset = sft_dataset.select(range(TRAIN_SIZE))
    test_dataset = sft_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + args.test_size))

    # -------------------------
    # Save
    # -------------------------
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Saved SFT train/test to {local_dir}")

    # Upload to HDFS
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

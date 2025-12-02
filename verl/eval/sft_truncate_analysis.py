import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import ipdb

# ===== 配置 =====
parquet_path = "/users/gwang16/Jillian/TinyZero/checkpoints/countdown_sft_truncate_think_maxthinktokens128/inference/global_step_284/countdown_sft_test_maxnewtokens1024_sft_generation.parquet"
model_name_or_path = "/users/gwang16/Jillian/TinyZero/checkpoints/countdown_sft_truncate_think_maxthinktokens128/global_step_284"  # 或你的 SFT 目录
max_input_length = 2048
max_new_tokens = 1024
near_max_threshold = 1024  # “接近 1024” 的下限，可以改成 800/950 等

# ===== 加载数据 & tokenizer =====
df = pd.read_parquet(parquet_path)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 推理时建议使用 left padding

prompt_key = "prompt"
response_key = "responses"

# ===== 工具函数 =====
def extract_prompt_text(p):
    """把 prompt 恢复成纯字符串（兼容 ndarray / list / dict / str）。"""
    if isinstance(p, np.ndarray):
        if p.dtype == object and len(p) > 0:
            p = p[0]

    if isinstance(p, list):
        if len(p) > 0 and isinstance(p[0], dict):
            return p[0].get("content", "")
        else:
            return str(p)
    elif isinstance(p, dict):
        return p.get("content", "")
    elif isinstance(p, str):
        return p
    else:
        return str(p)

def extract_response_text(r):
    """response 可能是 List[str] 或 str，只取第一个。"""
    if isinstance(r, list):
        if len(r) == 0:
            return ""
        return str(r[0])
    else:
        return str(r)

# ===== 加载数据 & tokenizer =====
df = pd.read_parquet(parquet_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total_rows = len(df)
cnt_valid = 0
cnt_near_max_no_answer = 0

for i in tqdm(range(total_rows), desc="Scanning samples"):
    prompt_obj = df.at[i, prompt_key]
    resp_obj = df.at[i, response_key]

    prompt_text = extract_prompt_text(prompt_obj)
    full_text = extract_response_text(resp_obj)

    if not isinstance(full_text, str) or len(full_text.strip()) == 0:
        continue  # 跳过空输出

    # 1) 统计 “新生成 token 数”
    # prompt 按 max_input_length truncate 一遍，模拟推理时的输入长度
    prompt_ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_input_length,
    )
    full_ids = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        truncation=False,          # 不截断，拿真实长度
    )
    len_prompt = len(prompt_ids)
    len_full = len(full_ids)
    new_len = max(len_full - len_prompt, 0)

    # 2) 在第一个 <think> 之后检查是否出现 </answer>
    tail = full_text
    idx = tail.find("Assistant: ")
    if idx != -1:
        tail = tail[idx + len("Assistant: "):]   # 只看第一个 <think> 之后的内容

    # ipdb.set_trace()
    has_answer = "</answer>" in tail

    # 3) 统计：既没有 </answer>，又接近 max_new_tokens
    cnt_valid += 1
    if (not has_answer) and (new_len >= near_max_threshold):
        cnt_near_max_no_answer += 1

print("Total valid samples:", cnt_valid)
print("Near-max & no </answer> samples:", cnt_near_max_no_answer)
if cnt_valid > 0:
    ratio = cnt_near_max_no_answer / cnt_valid
    print(
        f"Ratio: {ratio:.4f}  "
        f"(threshold>= {near_max_threshold}, max_new_tokens={max_new_tokens})"
    )
else:
    print("No valid samples.")

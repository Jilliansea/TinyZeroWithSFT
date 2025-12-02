# sft_generate.py
# 使用 SFT 后的 Qwen2.5-3B 模型，在 countdown 测试集上生成回答，并存成 parquet
#
# Config 示例（config/sft_generate.yaml）大致需要：
# model:
#   path: /path/to/sft_checkpoint  # HF 格式的 SFT 模型目录
#   name_or_path: Qwen/Qwen2.5-3B  # 仅在需要从 base 加载 tokenizer 时用
#   torch_dtype: bfloat16          # 或 float16 / float32
# data:
#   path: hdfs:///.../countdown_sft/test.parquet  # 原始 test.parquet
#   prompt_key: prompt
#   data_source_key: data_source
#   reward_model_key: reward_model
#   max_new_tokens: 256
#   batch_size: 8
#   temperature: 0.7
#   top_p: 0.9
# output:
#   path: hdfs:///.../countdown_sft/sft_outputs.parquet

import os
import math
import torch
import hydra
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import ipdb


def batch_iter(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]


def get_prompt(dataset, prompt_key, batch_idx):
    prompts = []
    for i in batch_idx:
        p = dataset.at[i, prompt_key]

        # Case 0: numpy array 包了一层
        if isinstance(p, np.ndarray):
            # 常见情况：array([{'content': '...', 'role': 'user'}], dtype=object)
            if p.dtype == object and len(p) > 0:
                p = p[0]   # 取出里面那个 dict/list

        # Case 1: [{"role": "user", "content": "..."}]
        if isinstance(p, list):
            if len(p) > 0 and isinstance(p[0], dict):
                text = p[0].get("content", "")
            else:
                text = str(p)

        # Case 2: {"role": "user", "content": "..."}
        elif isinstance(p, dict):
            text = p.get("content", "")

        # Case 3: 已经是纯字符串
        elif isinstance(p, str):
            text = p

        # Case 4: 其他类型（None / float / nan 等）
        else:
            text = str(p)

        prompts.append(text)
    return prompts


@hydra.main(config_path="config", config_name="sft_generation", version_base=None)
def main(cfg):
    # ipdb.set_trace()
    dataset = pd.read_parquet(cfg.data.path)

    prompt_key = cfg.data.prompt_key

    # 2. 加载 tokenizer & SFT 后模型
    # tokenizer 一般可以用 SFT 目录，也可以用 base 模型目录

    tokenizer_path = (
        cfg.model.path if getattr(cfg.model, "tokenizer_from_sft", True)
        else cfg.model.name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"         # ← 必须加
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_str = str(getattr(cfg.model, "torch_dtype", "bfloat16"))

    if dtype_str == "float16":
        torch_dtype = torch.float16
    elif dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 3. 准备推理参数
    max_new_tokens = cfg.data.max_new_tokens
    gen_batch_size = cfg.data.batch_size
    temperature = cfg.data.temperature
    top_p = cfg.data.top_p
    top_k = cfg.data.top_k
    all_indices = list(range(len(dataset)))

    # 新建一列，用于保存 SFT 生成结果（list[str]）
    dataset["responses"] = None

    # 4. 批量生成
    for batch_idx in tqdm(batch_iter(all_indices, gen_batch_size), total=len(all_indices)//gen_batch_size):
        prompts = get_prompt(dataset, prompt_key, batch_idx)

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=getattr(cfg.data, "max_input_length", 2048),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 只取生成的完整文本（包括 prompt + 输出），你后续 eval 时会把完整字符串扔进 reward 函数
        # ipdb.set_trace()
        for idx_in_batch, row_idx in enumerate(batch_idx):
            text = tokenizer.decode(outputs[idx_in_batch], skip_special_tokens=True)
            # 为了兼容 PPO 的 eval，这里存成 List[str]
            dataset.at[row_idx, "responses"] = [text]

    # 5. 保存到本地，再拷回 hdfs
    # ipdb.set_trace()
    out_local_dir = os.path.join(cfg.output.path, cfg.model.path.split("/")[-1])
    os.makedirs(out_local_dir, exist_ok=True)
    save_name = '_'.join([cfg.data.path.split("/")[-2], cfg.data.path.split("/")[-1].split(".")[0]])+'_maxnewtokens'+str(cfg.data.max_new_tokens)+'_sft_generation.parquet'
    out_local_path = os.path.join(out_local_dir, save_name)
    dataset.to_parquet(out_local_path)
    print(f"Saved SFT outputs to local file: {out_local_path}")


if __name__ == "__main__":
    main()

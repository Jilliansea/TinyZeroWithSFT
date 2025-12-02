# sft_eval.py
# 用 reward model 对 SFT 生成的 parquet 文件做离线评估，
# 接口形式和 PPO eval 脚本保持一致。

import os
import hydra
import numpy as np
import pandas as pd

from verl.utils.reward_score import math, gsm8k, multiply, countdown
import ipdb


def select_reward_fn(data_source: str):
    if data_source == "lighteval/MATH":
        return math.compute_score
    elif data_source == "gsm8k":
        return gsm8k.compute_score
    elif data_source == "multiply":
        return multiply.compute_score
    elif data_source == "countdown" or data_source == "countdown_sft":
        return countdown.compute_score_for_analysis
    else:
        raise NotImplementedError(f"Unknown data_source: {data_source}")


@hydra.main(config_path="config", config_name="sft_evaluation", version_base=None)
def main(config):
    """
    config.data.path: hdfs or local path to SFT outputs parquet
    config.data.prompt_key: usually "prompt"
    config.data.response_key: usually "response"
    config.data.data_source_key: usually "data_source"
    config.data.reward_model_key: usually "reward_model"
    config.output_dir: where to save passes/not_passes parquet
    """
    local_path = config.data.path
    dataset = pd.read_parquet(local_path)

    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    # 预留字段
    dataset = dataset.assign(reason=None, max_score=None)

    os.makedirs(config.output_dir, exist_ok=True)

    passes = 0
    passes_dataset = []
    not_passes_dataset = []

    total = len(dataset)

    for i in range(total):
        # response_lst 既可能是 List[str]，也可能是单个 str
        r = responses[i]
        # ipdb.set_trace()

        if isinstance(r, list):
            response_lst = r
        elif isinstance(r, np.ndarray):
            response_lst = r.tolist()
        else:
            response_lst = [r]
        # ipdb.set_trace()

        data_source = data_sources[i]
        reward_fn = select_reward_fn(data_source)

        reward_data = reward_model_data[i]
        ground_truth = reward_data["ground_truth"]

        score_lst = []
        reason_lst = []

        for resp in response_lst:
            score, reason = reward_fn(solution_str=resp, ground_truth=ground_truth)
            # ipdb.set_trace()
            score_lst.append(score)
            reason_lst.append(reason)

        max_score = float(np.max(score_lst))
        max_reason = reason_lst[int(np.argmax(score_lst))]

        dataset.at[i, "reason"] = max_reason
        dataset.at[i, "max_score"] = max_score

        if max_score == 1.0:
            passes += 1
            passes_dataset.append(dataset.iloc[i])
        else:
            not_passes_dataset.append(dataset.iloc[i])

    passes_dataset = pd.DataFrame(passes_dataset)
    not_passes_dataset = pd.DataFrame(not_passes_dataset)
    passes_dataset.reset_index(drop=True, inplace=True)
    not_passes_dataset.reset_index(drop=True, inplace=True)

    val_num = 1  # 对应 pass@1
    passes_path = os.path.join(config.output_dir, f"{config.data.path.split('/')[-1].split('.')[0]}_passes_dataset_{val_num}.parquet")
    not_passes_path = os.path.join(config.output_dir, f"{config.data.path.split('/')[-1].split('.')[0]}_not_passes_dataset_{val_num}.parquet")

    passes_dataset.to_parquet(passes_path)
    not_passes_dataset.to_parquet(not_passes_path)

    print(f"total: {total}")
    print(f"pass@{val_num}: {passes / total:.4f}")
    print(f"saved passes to: {passes_path}")
    print(f"saved not_passes to: {not_passes_path}")


if __name__ == "__main__":
    main()

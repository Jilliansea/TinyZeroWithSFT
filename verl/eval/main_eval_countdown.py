# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import os
import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import math, gsm8k, multiply, countdown
import pandas as pd
import numpy as np
import ipdb



def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        return math.compute_score
    elif data_source == 'countdown':
        return countdown.compute_score_for_analysis
    else:
        raise NotImplementedError


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_local_path_from_hdfs(config.data.path)
    dataset = pd.read_parquet(local_path)
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    dataset = dataset.assign(reason=None, max_score=None)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    passes = 0
    passes_dataset = []
    not_passes_dataset = []

    total = len(dataset)
    val_num = 1

    for i in range(total):
        response_lst = responses[i]
        if isinstance(response_lst, np.ndarray):
            response_lst = response_lst.tolist()
        val_num = len(response_lst)
        data_source = data_sources[i]
        # select reward score based on data_source
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        reason_lst = []
        for r in response_lst:
            # ipdb.set_trace()
            score, reason = reward_fn(solution_str=r, ground_truth=ground_truth)
            score_lst.append(score)
            reason_lst.append(reason)

        max_score = np.max(score_lst)
        max_reason = reason_lst[np.argmax(score_lst)]
        dataset.at[i, 'reason'] = max_reason
        dataset.at[i, 'max_score'] = max_score

        # 把 passes样本和not passes样本分别保存到两个不同的parquet文件
        if max_score == 1:
            passes += 1
            passes_dataset.append(dataset.iloc[i])
        else:
            not_passes_dataset.append(dataset.iloc[i])

    
    passes_dataset = pd.DataFrame(passes_dataset)
    not_passes_dataset = pd.DataFrame(not_passes_dataset)
    passes_dataset.reset_index(drop=True, inplace=True)
    not_passes_dataset.reset_index(drop=True, inplace=True)
    passes_dataset.to_parquet(os.path.join(output_dir, f'passes_dataset_{val_num}.parquet'))
    not_passes_dataset.to_parquet(os.path.join(output_dir, f'not_passes_dataset_{val_num}.parquet'))

    print(f'pass@{val_num}: {passes / total}')
    # print(f'pass@5: {passes / total}')


if __name__ == '__main__':
    main()

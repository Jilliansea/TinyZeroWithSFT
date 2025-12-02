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

import os, sys
import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import math, gsm8k, multiply, countdown
import pandas as pd
import numpy as np
import ipdb
import collections
from collections import Counter
# 固定随机数种子，保证结果可复现
import random
random.seed(42)
from fractions import Fraction
from itertools import combinations


def solve_with_all_numbers(target, numbers):
    """
    必须使用给定 numbers 的所有数字，判断是否能得到 target。
    能解则返回一个表达式字符串，否则返回 None。
    """

    target_frac = Fraction(target)

    # 每个元素表示 (数值, 表达式)
    values = [(Fraction(n), str(n)) for n in numbers]

    def helper(vals):
        # vals: list[(Fraction, expr_str)]
        if len(vals) == 1:
            if vals[0][0] == target_frac:
                return vals[0][1]
            else:
                return None

        # 从当前列表中任选两个数做运算
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a_val, a_expr = vals[i]
                b_val, b_expr = vals[j]

                # 剩余没用的数
                rest = [vals[k] for k in range(len(vals)) if k != i and k != j]

                candidates = []

                # 加法（交换律可以去重，但这里先不做优化）
                candidates.append((a_val + b_val, f"({a_expr} + {b_expr})"))

                # 减法（顺序都试）
                candidates.append((a_val - b_val, f"({a_expr} - {b_expr})"))
                candidates.append((b_val - a_val, f"({b_expr} - {a_expr})"))

                # 乘法
                candidates.append((a_val * b_val, f"({a_expr} * {b_expr})"))

                # 除法（保证不除以 0）
                if b_val != 0:
                    candidates.append((a_val / b_val, f"({a_expr} / {b_expr})"))
                if a_val != 0:
                    candidates.append((b_val / a_val, f"({b_expr} / {a_expr})"))

                # 递归尝试
                for new_val, new_expr in candidates:
                    new_vals = rest + [(new_val, new_expr)]
                    res = helper(new_vals)
                    if res is not None:
                        return res

        return None

    return helper(values)


def solve_countdown(target, numbers, allow_subset=True):
    """
    allow_subset=True 表示可以只用 numbers 的子集（不少于2个），
    False 表示必须用完所有 numbers。
    """

    if not allow_subset:
        return solve_with_all_numbers(target, numbers)

    # 允许用任意长度 >= 2 的子集
    for r in range(2, len(numbers) + 1):
        for idxs in combinations(range(len(numbers)), r):
            subset = [numbers[i] for i in idxs]
            expr = solve_with_all_numbers(target, subset)
            if expr is not None:
                return expr

    return None


def check_row_has_solution(row, allow_subset=True):
    target = int(row["target"])
    numbers = row["nums"]   # 假设本身就是 list[int]
    expr = solve_countdown(target, numbers, allow_subset=allow_subset)
    return pd.Series({
        "has_solution": expr is not None,
        "expression": expr
    })


def batch_check_parquet(df, allow_subset=True):

    # 如果 numbers 存成了字符串，例如 "[80, 35, 74, 7]"，可以先转一下：
    # import ast
    # df["numbers"] = df["numbers"].apply(ast.literal_eval)

    result = df.apply(
        lambda row: check_row_has_solution(row, allow_subset=allow_subset),
        axis=1
    )

    df["has_solution"] = result["has_solution"]
    df["expression"] = result["expression"]
    return df



def main(file_path, display=False):
    total_sample_count = 1024
    file_local_path = copy_local_path_from_hdfs(file_path)
    dataset = pd.read_parquet(file_local_path)
    if 'responses' in dataset.columns:
        responses = dataset['responses']
    else:
        responses = None
        
    targets = dataset['target']
    numbers = dataset['nums']
    reasons = dataset['reason']

    # ====================== 数据集基本信息分析 ======================
    # 样本数量
    sample_count = dataset.shape[0]
    print(f" |----- Sample count: {sample_count} -----|")

    # 检查是否有解 
    dataset = batch_check_parquet(
        dataset,
        allow_subset=False  # 是否允许只用数字的子集
    )
    # 统计没有解的样本数量
    no_solution_count = dataset[dataset['has_solution'] == False].shape[0]
    print(f" |----- No solution count: {no_solution_count} -----|")

    #  统计加、减、乘、除各自出现的次数和比例（每个表达式中一种运算符仅计算一次）
    add_count = 0
    sub_count = 0
    multiply_count = 0
    divide_count = 0
    multiply_divide_count = 0
    add_sub_count = 0
    '''
    for expression in dataset['expression']:
        if '+' in expression:
            add_count += 1
        if '-' in expression:
            sub_count += 1
        if '*' in expression:
            multiply_count += 1
        if '/' in expression:
            divide_count += 1
    print(f" |----- Add count: {add_count} -----|")
    print(f" |----- Sub count: {sub_count} -----|")
    print(f" |----- Multiply count: {multiply_count} -----|")
    print(f" |----- Divide count: {divide_count} -----|")
    print(f" |----- Add ratio: {add_count / dataset.shape[0]} -----|")
    print(f" |----- Sub ratio: {sub_count / dataset.shape[0]} -----|")
    print(f" |----- Multiply ratio: {multiply_count / dataset.shape[0]} -----|")
    print(f" |----- Divide ratio: {divide_count / dataset.shape[0]} -----|")
    '''
    for expression in dataset['expression']:
        if '*' in expression or '/' in expression:
            multiply_divide_count += 1
        else:
            add_sub_count += 1
    print(f" |----- Multiply divide count: {multiply_divide_count} -----|")
    print(f" |----- Multiply divide ratio: {multiply_divide_count / total_sample_count} -----|")
    print(f" |----- Add sub count: {add_sub_count} -----|")
    print(f" |----- Add sub ratio: {add_sub_count / total_sample_count} -----|")

    # ====================== 分析不同reason的分布 ======================
    reason_counter = Counter(reasons)
    print(f"Reason distribution: {reason_counter}")
    # 统计不同reason的比例
    reason_ratio = {}
    for reason in reason_counter.keys():
        reason_ratio[reason] = reason_counter[reason] / total_sample_count
    print(f"Reason ratio: {reason_ratio}")

    # 每个reason随机抽取50个，显示equation和ground truth
    for reason in reason_counter.keys():
        print(f" ===== Reason: {reason} ===== ")
        reason_indices = np.where(reasons == reason)[0]
        random_indices = np.random.choice(reason_indices, min(20, len(reason_indices)), replace=False)
        for i in random_indices:
            print(f"Target: {targets[i]}, numbers: {numbers[i]}")
            if responses is not None:
                print(f"Response: {responses[i]}")
            print(f"Expression: {dataset['expression'][i]}\n")


if __name__ == '__main__':
    file_path = sys.argv[1]
    main(file_path=file_path, display=True)
    # pass_file = './checkpoints/TinyZero/countdown-fixed-p2p-disabled_1115/inference/eval/passes_dataset_1.parquet'
    # main(file_path=pass_file, display=True)

    # not_pass_file = './checkpoints/TinyZero/countdown-fixed-p2p-disabled_1115/inference/eval/not_passes_dataset_1.parquet'
    # main(file_path=not_pass_file, display=True)

    # train_file = '/users/gwang16/Jillian/TinyZero/data/countdown'
    # main(file_path=train_file)

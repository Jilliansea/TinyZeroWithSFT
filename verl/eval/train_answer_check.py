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
import re


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


def is_valid_equation(eq: str):
    """
    只允许数字、空格、括号和 + - * / 四种运算符。
    如果出现其他字符或非法运算符，则视为非法。
    """

    # 去掉空白
    eq_clean = eq.strip()

    # 检查是否包含非法字符 (允许数字、空格、括号、+ - * /)
    if not re.fullmatch(r"[0-9+\-*/() ]+", eq_clean):
        return False

    # 检查是否存在非法多字符操作符，如 **、//、++、-- 等
    illegal_patterns = [r"\*\*", r"//", r"\+\+", r"--"]
    for p in illegal_patterns:
        if re.search(p, eq_clean):
            return False

    return True


def main(file_path, display=False):
    file_local_path = copy_local_path_from_hdfs(file_path)
    dataset = pd.read_parquet(file_local_path)
    if 'equation' in dataset.columns:
        equations = dataset['equation']
    else:
        equations = None
    

    invalid_samples = []
    for eq in equations:
        print(f"eq: {eq}")
        is_valid = is_valid_equation(eq)
        # ipdb.set_trace()
        if not is_valid:
            invalid_samples.append(eq)
            print(f"非法样本: {eq}")

    # 统计
    total = len(equations)
    invalid_cnt = len(invalid_samples)
    invalid_ratio = invalid_cnt / total if total > 0 else 0.0

    print("非法样本:")
    for s in invalid_samples:
        print("  ", s)

    print("\n统计：")
    print(f"总样本数: {total}")
    print(f"非法样本数: {invalid_cnt}")
    print(f"非法比例: {invalid_ratio:.4f}")



if __name__ == '__main__':
    train_file = '/users/gwang16/Jillian/TinyZero/data/countdown_sft/train.parquet'
    main(file_path=train_file)

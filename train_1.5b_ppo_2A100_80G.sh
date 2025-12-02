#!/bin/bash
ray stop --force || true
rm -rf /tmp/ray

# 应用环境修复
source ./fix_environment.sh  

# 训练配置
export N_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export BASE_MODEL="Qwen/Qwen2.5-1.5B"
export DATA_DIR="data/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-fixed-p2p-disabled_1115
export VLLM_ATTENTION_BACKEND=XFORMERS

echo "=== Starting Training with P2P Disabled ==="
echo "This avoids GPU communication issues by using system memory for transfers"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Rollout TP size: $ROLLOUT_TP_SIZE"

# 运行训练脚本
bash ./scripts/train_ppo.sh

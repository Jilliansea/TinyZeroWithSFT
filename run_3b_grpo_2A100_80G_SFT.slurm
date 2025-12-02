#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --time=720:00:00             # GPU,无时间限制or >128h

#SBATCH --nodelist=str-gpu14        # 节点信息
#SBATCH --mem=200gb                 # 根据节点修改

#SBATCH --gres=gpu:2
#SBATCH --output=logs/slurm/slurm-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%A_%a.err

source ~/.bashrc
conda activate zero121

echo "=== SLURM ENV ==="
echo "HOSTNAME          = $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi

bash train_3b_grpo_sft.sh

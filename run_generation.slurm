#!/bin/bash
#SBATCH --partition=Nebula_GPU
#SBATCH --time=48:00:00             # GPU,无时间限制or >128h

#SBATCH --nodelist=str-gpu11        # 节点信息
#SBATCH --mem=200gb                 # 根据节点修改

#SBATCH --gres=gpu:1

#SBATCH --output=logs/slurm/slurm-%A_%a.out
#SBATCH --error=logs/slurm/slurm-%A_%a.err

source ~/.bashrc

conda activate zero121

bash scripts/inference_ppo.sh

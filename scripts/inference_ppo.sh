#!/bin/bash
model_path=$1
output_path=$2
python3 -m verl.eval.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=./data/countdown/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=${output_path} \
    model.path=${model_path} \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8

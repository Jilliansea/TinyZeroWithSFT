#!/bin/bash
# model_path: checkpoints/countdown-sft-qwen2.5-3b/global_step_142
model_path=$1
python3 -m verl.eval.sft_generation \
    data.path=./data/countdown_sft/test.parquet \
    data.prompt_key=prompt \
    data.batch_size=64 \
    data.temperature=1.0 \
    data.top_p=0.7 \
    data.top_k=50 \
    data.max_input_length=2048 \
    data.max_new_tokens=1024 \
    model.path=${model_path} \
    model.name_or_path=Qwen/Qwen2.5-3B \
    model.tokenizer_from_sft=True \
    output.path=./checkpoints/countdown_sft_truncate_think_maxthinktokens128/inference
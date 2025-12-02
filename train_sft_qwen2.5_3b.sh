set -x
# 保存路径
local_path=/users/gwang16/Jillian/TinyZero/checkpoints/countdown_sft

nproc_per_node=$1

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=./data/countdown_sft/train.parquet \
    data.val_files=./data/countdown_sft/test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.train_batch_size=128 \
    data.micro_batch_size=8 \
    data.max_length=2048 \
    data.truncation=right \
    model.partial_pretrain=Qwen/Qwen2.5-3B \
    trainer.default_local_dir=$local_path \
    trainer.project_name=countdown-sft \
    trainer.experiment_name=countdown-sft-qwen2.5-3b \
    trainer.total_epochs=2 \
    trainer.logger=['wandb'] \
    trainer.save_every_steps=1 \
    trainer.val_every_steps=1 \
    optim.lr=5e-6


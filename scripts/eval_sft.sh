data_path=$1
output_dir=$2
python3 -m verl.eval.sft_eval \
    data.path=${data_path} \
    output_dir=${output_dir}

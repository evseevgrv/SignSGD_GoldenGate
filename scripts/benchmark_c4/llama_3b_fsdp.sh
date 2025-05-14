#!/bin/bash

export OMP_NUM_THREADS=8
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

export WANDB_DISABLED="false"
export WANDB_PROJECT="galore_llama_3b_TEST"
model_config=configs/llama_3b.json
save_dir_prefix=3b

n_of_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
command="torchrun --standalone --nproc_per_node $n_of_visible_devices torchrun_main_fsdp.py --model_config $model_config --save_dir_prefix $save_dir_prefix "

# command+=" --wandb_name_prefix TEST"

command+=" --optimizer adam"
command+=" --dtype fp32"
amp=1
amp_arg=$( [ "$amp" = 0 ] && echo "--no-amp" || echo "--amp" )
command+=" $amp_arg"
command+=" --total_batch_size 512"
command+=" --batch_size 32"
command+=" --scheduler cosine"
command+=" --scheduler_cycle_length 150000"
command+=" --weight_decay 0.00"
command+=" --lr 0.00016"
command+=" --grad_clipping 1.0"

# seed
command+=" --seed 0"

# # optimizer args
command+=" --eps 1e-8"
command+=" --beta1 0.9"
command+=" --beta2 0.999"

# other
command+=" --eval_batch_size 256"
command+=" --warmup_steps 15000"
command+=" --eval_every 2"
command+=" --save_every 2000"
command+=" --num_training_steps 150000"

# metrics
run_final_eval=0
compute_stable_rank=0
run_final_eval_arg=$( [ "$run_final_eval" = 0 ] && echo "--no-run_final_eval" || echo "--run_final_eval" )
compute_stable_rank_arg=$( [ "$compute_stable_rank" = 0 ] && echo "--no-compute_stable_rank" || echo "--compute_stable_rank" )

# Append special arguments to command
command+=" $run_final_eval_arg"
command+=" $compute_stable_rank_arg"
# command+=" --single_gpu"
# command+=" --local_train_data"

# export WANDB_API_KEY=

eval $command

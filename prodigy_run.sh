#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=0
 
export WANDB_DISABLED="false"
export WANDB_PROJECT="comparison"
model_config=configs/llama_130m.json
save_dir_prefix=130m
 
n_of_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
command="torchrun --standalone --nproc_per_node $n_of_visible_devices torchrun_main.py --model_config $model_config --save_dir_prefix $save_dir_prefix "
 
# command+=" --wandb_name_prefix TEST"
 
compile=1
compile_arg=$( [ "$compile" = 0 ] && echo "--no-compile" || echo "--compile" )
command+=" $compile_arg"
 
hf_model=0
hf_model_arg=$( [ "$hf_model" = 0 ] && echo "--no-use_hf_model" || echo "--use_hf_model" )
command+=" $hf_model_arg"
 
command+=" --optimizer prodigy"
command+=" --dtype fp32"
amp=1
amp_arg=$( [ "$amp" = 0 ] && echo "--no-amp" || echo "--amp" )
command+=" $amp_arg"
command+=" --total_batch_size 512"
command+=" --batch_size 128"
command+=" --scheduler cosine"
command+=" --scheduler_cycle_length 100000"
command+=" --weight_decay 0.001"
command+=" --lr 1e-2"
command+=" --grad_clipping 1.0"
 
# parameter free specific
command+=" --l_inf 100.0"
command+=" --lower_bound 0.0"
command+=" --clamp_level 0.001"
# command+=" --d_0 1.0"
command+=" --update_gap 20"
 
command+=" --proj_embeds"
command+=" --proj_norms"
# command+=" --proj_logits"
 
# seed
command+=" --seed 0"
 
command+=" --eps 1e-8"
command+=" --beta1 0.9"
command+=" --beta2 0.999"
command+=" --momentum 0.9"
# other
command+=" --eval_batch_size 256"
command+=" --warmup_steps 10000"
# command+=" --warmup_steps 0"
command+=" --eval_every 2000"
command+=" --save_every 1000"
command+=" --num_training_steps 100000"
 
# metrics
 
run_final_eval=1
 
compute_stable_rank=0
run_final_eval_arg=$( [ "$run_final_eval" = 0 ] && echo "--no-run_final_eval" || echo "--run_final_eval" )
compute_stable_rank_arg=$( [ "$compute_stable_rank" = 0 ] && echo "--no-compute_stable_rank" || echo "--compute_stable_rank" )
 
# Append special arguments to command
command+=" $run_final_eval_arg"
# command+=" --single_gpu"
command+=" --local_train_data"
# command+=" --workers 1"
 
export WANDB_API_KEY=$(cat /slot/sandbox/d/secret/*)
 
eval $command

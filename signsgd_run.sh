#!/bin/bash
 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 
export WANDB_DISABLED="false"
export WANDB_PROJECT="sgd_with_adam_350"
model_config=configs/llama_350m.json
save_dir_prefix=350m
 
n_of_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
command="torchrun --standalone --nproc_per_node $n_of_visible_devices torchrun_main.py --model_config $model_config --save_dir_prefix $save_dir_prefix "
 
# command+=" --wandb_name_prefix TEST"
 
command+=" --optimizer sgd_with_adam"
command+=" --dtype fp32"
amp=1
amp_arg=$( [ "$amp" = 0 ] && echo "--no-amp" || echo "--amp" )
command+=" $amp_arg"
command+=" --total_batch_size 512"
command+=" --batch_size 64"
command+=" --scheduler cosine"
command+=" --scheduler_cycle_length 100000"
command+=" --weight_decay 0.01"
command+=" --lr 0.001"
# command+=" --grad_clipping 1.0"
 
# proj specific
command+=" --proj_params_lr_scale 1.78"
command+=" --update_gap 200"
command+=" --density 0.0"
reset_statistics=1
reset_statistics_arg=$( [ "$reset_statistics" = 0 ] && echo "--no-reset_statistics" || echo "--reset_statistics" )
command+=" $reset_statistics_arg"
command+=" --inactive_update_rule no"
command+=" --inactive_lr_scale 1.0"
command+=" --proj_embeds"
command+=" --proj_norms"
# command+=" --proj_logits"
 
# seed
command+=" --seed 0"
 
# # optimizer args
# command+=" --nesterov"
command+=" --momentum 0.0"
command+=" --sgd_sign_update"
# command+=" --sign_norm"
command+=" --eps 1e-8"
command+=" --beta1 0.9"
command+=" --beta2 0.999"
 
# other
command+=" --eval_batch_size 256"
command+=" --warmup_steps 10000"
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
 
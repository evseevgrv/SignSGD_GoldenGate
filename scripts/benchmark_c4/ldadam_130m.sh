export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=6,7

export WANDB_DISABLED="false"
export WANDB_PROJECT="galore_llama_130m"
model_config=configs/llama_130m.json
save_dir_prefix=130m

n_of_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
command="torchrun --standalone --nproc_per_node $n_of_visible_devices torchrun_main.py --model_config $model_config --save_dir_prefix $save_dir_prefix "
# command+=" --debug"
# command+=" --no-wandb"
# command+=" --debug_train"
# command+=" --debug_print"

compile=0
compile_arg=$( [ "$compile" = 0 ] && echo "--no-compile" || echo "--compile" )
command+=" $compile_arg"

hf_model=0
hf_model_arg=$( [ "$hf_model" = 0 ] && echo "--no-use_hf_model" || echo "--use_hf_model" )
command+=" $hf_model_arg"

# command+=" --attn_implementation flash_attention_2"

# command+=" --fsdp"
# command+=" --mp_policy_param float32"
# command+=" --mp_policy_reduce float32"
# command+=" --mp_policy_buffer float32"
# command+=" --cpu_offload"

# command+=" --wandb_name_prefix ddp"

command+=" --optimizer ldadam"
command+=" --dtype bf16"
amp=0
amp_arg=$( [ "$amp" = 0 ] && echo "--no-amp" || echo "--amp" )
command+=" $amp_arg"
command+=" --total_batch_size 512"
command+=" --batch_size 256"
# command+=" --max_length 256"
command+=" --scheduler cosine"
command+=" --scheduler_cycle_length 20000"
command+=" --weight_decay 0.00"
command+=" --lr 0.001"
# command+=" --grad_clipping 1.0"

# proj specific
# command+=" --proj_params_lr_scale 1.0"
command+=" --update_gap 200"
command+=" --density 0.25"
# reset_statistics=1
# reset_statistics_arg=$( [ "$reset_statistics" = 0 ] && echo "--no-reset_statistics" || echo "--reset_statistics" )
# command+=" $reset_statistics_arg"
# command+=" --inactive_update_rule sign_sgd"
# command+=" --inactive_lr_scale 1.0"
# command+=" --proj_embeds"
# command+=" --proj_norms"
# command+=" --proj_logits"

# # galore specific
# command+=" --proj_side std"
# command+=" --proj_type svd"

# apollo specific

# # coord specific
# command+=" --coord_choice randk"

# # block specific
# command+=" --block_order descending"

# # apollo specific
# command+=" --scale_type channel"
# command+=" --proj_side std"
# command+=" --apollo_scale 1.0"

# apollo specific
command+=" --ldadam_rho 0.908"
command+=" --proj_side std"
command+=" --ldadam_proj_method power_iteration"
ldadam_error_feedback=1
ldadam_error_feedback_arg=$( [ "$ldadam_error_feedback" = 0 ] && echo "--no-ldadam_error_feedback" || echo "--ldadam_error_feedback" )
command+=" $ldadam_error_feedback_arg"

# seed
command+=" --seed 0"

# # optimizer args
command+=" --eps 1e-8"
command+=" --beta1 0.908"
command+=" --beta2 0.99"

# other
command+=" --eval_batch_size 256"
command+=" --warmup_steps 2000"
command+=" --eval_every 1000"
command+=" --save_every 1000"
command+=" --num_training_steps 20000"

# metrics
run_final_eval=0
run_final_eval_arg=$( [ "$run_final_eval" = 0 ] && echo "--no-run_final_eval" || echo "--run_final_eval" )

# Append special arguments to command
command+=" $run_final_eval_arg"
command+=" --single_gpu"
# command+=" --local_train_data"
# command+=" --workers 1"
# command+=" --final_save"

export WANDB_API_KEY=$(cat /slot/sandbox/d/secret/*)

command+=" > apollo_130m.txt"

eval $command
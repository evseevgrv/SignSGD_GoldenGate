export HF_HOME=/mnt/LLM
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=8

python finetune.py \
  --base_model 'meta-llama/Llama-3.2-1B' \
  --data_path 'commonsense_15k.json' \
  --output_dir './trained_models/llama-lora' \
  --save_step 10 \
  --eval_step 10 \
  --batch_size 16 \
  --micro_batch_size 8 \
  --num_epochs 3 \
  --learning_rate 3e-2 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]' \
  --optimizer_name adamw \
  --compile 0 \
  --adapter_name lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --max_steps 12 \
  # --attn_implementation flash_attention_2 \
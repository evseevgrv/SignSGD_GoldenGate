export HF_HOME=/mnt/LLM
export OMP_NUM_THREADS=8

mkdir ./result/test_1b_model_fira -p

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset boolq \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/boolq.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset piqa \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/piqa.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset social_i_qa \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/social_i_qa.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset hellaswag \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/hellaswag.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset winogrande \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/winogrande.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset ARC-Easy \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/ARC-Easy.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset ARC-Challenge \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/ARC-Challenge.txt'

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter no \
    --dataset openbookqa \
    --batch_size 32 \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --lora_weights './trained_models/llama-fira' | tee -a './result/test_1b_model_fira/openbookqa.txt'
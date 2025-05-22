# run_script.sh (обновленная версия)
#!/bin/bash
export CUDA_LAUNCH_BLOCKING=2

wandb login

DEVICE=0
EXPERIMENTS_DIR="experiments"

mkdir -p $EXPERIMENTS_DIR

declare -A methods=( 
    ["adamlike"]="--lr 0.001" 
)

for method in "${!methods[@]}"; do
    echo "Running $method with params: ${methods[$method]}"
    python run_svrg_reb_swin.py \
        --method $method \
        --device $DEVICE \
        ${methods[$method]}
    
    sleep 10
done

echo "All experiments completed! Results in $EXPERIMENTS_DIR"
python plot.py

#!/bin/bash
#SBATCH --job-name=deepfake_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=defaultq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepfake_env

cd ~/Zero-Trust-Deepfake-Voice-Defense

python scripts/evaluate.py \
    --config configs/model_config.yaml \
    --data-dir data/release_in_the_wild \
    --dataset in_the_wild \
    --checkpoint models/best_checkpoint.pt

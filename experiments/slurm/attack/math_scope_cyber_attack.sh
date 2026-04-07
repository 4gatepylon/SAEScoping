#!/bin/bash
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=math-in-cyber-attack
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=48G
#SBATCH --requeue

nvidia-smi

source ~/.bashrc
conda activate sae
cd ~/sae-filters/SAEScoping

python experiments/script_scoping_pipeline_stemqa.py \
    --train-domain math --attack-domain cyber --stage attack \
    --batch-size 2 --checkpoint experiments/outputs_scoping/math/recover/checkpoint-3000

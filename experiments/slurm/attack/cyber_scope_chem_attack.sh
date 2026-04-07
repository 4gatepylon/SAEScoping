#!/bin/bash
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=cyber-in-chem-attack
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --requeue

nvidia-smi

source ~/.bashrc
conda activate sae
cd ~/sae-filters/SAEScoping

python experiments/script_scoping_pipeline_stemqa.py \
    --train-domain cyber --attack-domain chemistry --stage attack \
    --batch-size 2 --checkpoint experiments/outputs_scoping/cyber/recover/checkpoint-3000

#!/bin/bash
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=chem-in-math-attack
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --requeue

nvidia-smi

source ~/.bashrc
conda activate sae
cd ~/sae-filters/SAEScoping

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain math --attack-domain chemistry --stage attack \
    --checkpoint outputs_scoping/math/recover/checkpoint-3000

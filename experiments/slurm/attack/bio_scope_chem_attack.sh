#!/bin/bash
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=bio-in-chem-attack
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --requeue

nvidia-smi

source ~/.bashrc
source /data/aruna_sankaranarayanan/miniconda3/etc/profile.d/conda.sh
conda info --envs
conda activate sae
cd ~/sae-filters/SAEScoping

python experiments/script_scoping_pipeline_stemqa.py \
    --train-domain biology --attack-domain chemistry --stage attack \
    --batch-size 2 --checkpoint experiments/outputs_scoping/biology/recover/checkpoint-3000

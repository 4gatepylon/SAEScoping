#!/bin/bash
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=chem-in-bio-attack
#SBATCH -o logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --requeue

nvidia-smi

source ~/.bashrc
conda activate sae
cd ~/sae-filters/SAEScoping

python experiments/script_scoping_pipeline_stemqa.py \
    --train-domain chemistry --attack-domain biology --stage attack \
    --batch-size 2 --checkpoint experiments/outputs_scoping/chemistry/recover/checkpoint-3000

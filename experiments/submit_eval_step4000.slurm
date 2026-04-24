#!/bin/bash
#SBATCH --job-name=eval_step4000
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_step4000_%j.out
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python experiments/eval_final_judge_step4000.py

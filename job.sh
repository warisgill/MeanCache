#!/bin/bash
 
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_normal_q
#SBATCH --account=semcache
 
module reset
module load Anaconda3

conda init

source ~/.bashrc

source activate llm

# python generate_scripts.py

# bash _run.sh
python loop.py





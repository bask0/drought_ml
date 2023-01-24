#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --partition=gpu
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=0-02:00:00
#SBATCH --output="./logs/version_%A/slurm.out"

# activate conda env
eval "$(conda shell.bash hook)"
source activate dml

# debugging flags (optional)
export NCCL_DEBUG=ERROR
export PYTHONFAULTHANDLER=1

module load cuda/11.6

# run script from above
srun python ./project/run_model.py --overfit

#!/bin/bash
# job.sbatch
#
#SBATCH -J dataprep # A single job name for the array
#SBATCH -p work # partition
#SBATCH --cpus-per-task=5 # 5 cores
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH -t 0-2:00 # Running time of 2 hours
#SBATCH -o preprocessing/logs/dataprep_%A_%a.out # Standard output
#SBATCH -e preprocessing/logs/dataprep_%A_%a.err # Standard error

source files_array

FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}

python preprocessing/cube_harmonize.py \
      -f $FILENAME \
      -o /Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr

#!/bin/bash
# job.sbatch
#
#SBATCH -J dataprep # A single job name for the array
#SBATCH -p big # partition
#SBATCH --cpus-per-task=5 # 5 cores
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH -t 0-10:00 # Running time of 10 hours
#SBATCH -o preprocessing/logs/dataprep_%A_%a.out # Standard output

echo "Running python command:"
echo $@
echo ""

$@

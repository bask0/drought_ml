#!/bin/bash

# Sort does sort first sort by month, then by year. This separates consecutive months of the same variable
# to avoid race condition (if chunks overlap months). Don't use less than four years because this could still
# cause race condition.
FILES=($(ls -1 /Net/Groups/BGI/scratch/bkraft/drought_data/preproc/*.static.1460.1140.nc))
FILES+=($(ls -1 /Net/Groups/BGI/scratch/bkraft/drought_data/preproc/*{2002..2021}.nc | sort -t "." -k5 -k6))

NUMFILES=${#FILES[@]}
ARRNUMFILES=$(($NUMFILES - 1))

rm -rf preprocessing/logs/ 
mkdir -p preprocessing/logs/

echo ">>> Preparing cube..."

python preprocessing/cube_harmonize.py \
  --create \
  -o /Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr

echo ">>> Processing ${NUMFILES} files..."

declare -p FILES | sed 's/ -[aA]/&g/' > files_array

if [ $ARRNUMFILES -ge 0 ]
then
    sbatch --wait --array=0-$ARRNUMFILES%20 --partition=big preprocessing/cube_add_file.sbatch
else
    echo ">>> No files to process"
fi

wait

rm files_array

echo ">>> Computing mask..."

python preprocessing/compute_mask.py \
  -p /Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr

echo ">>> Computing anomalies..."

python preprocessing/cube_harmonize.py \
      -o /Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr \
      --anomalies fvc

python preprocessing/cube_harmonize.py \
      -o /Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr \
      --anomalies lst

echo ">>> Done"

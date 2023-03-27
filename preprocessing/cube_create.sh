#!/bin/bash

cube_path="/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr"
max_num_par=60

# Sort does sort first by month, then by year. This separates consecutive months of the same variable
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
  -o $cube_path

echo ">>> Processing ${NUMFILES} files..."

declare -p FILES | sed 's/ -[aA]/&g/' > files_array

if [ $ARRNUMFILES -ge 0 ]
then
    sbatch -J dwrite --wait --array=0-$ARRNUMFILES%$max_num_par --partition=big preprocessing/cube_add_file.sbatch
else
    echo ">>> No files to process"
      rm files_array
      exit 1
fi

wait

rm files_array

echo ">>> Postprocessing..."

command="python preprocessing/compute_mask.py -p $cube_path"
sbatch -J mask --wait preprocessing/slurm_python_call.sbatch $command

wait

# Converting ER55-land cummmulative fluxes to fluxes

command="python preprocessing/fix_era_vars.py -p $cube_path -v tp -n 30"
JOBID1=$(sbatch -J fixtp --parsable --cpus-per-task=30 preprocessing/slurm_python_call.sbatch $command)

command="python preprocessing/fix_era_vars.py -p $cube_path -v ssrd -n 30"
JOBID2=$(sbatch -J fixssrd --parsable --cpus-per-task=30 preprocessing/slurm_python_call.sbatch $command)

# Computing feature seasonalities...

s_t2m="python preprocessing/compute_ano_and_msc.py -p $cube_path -v t2m --msc_only"
s_tp="python preprocessing/compute_ano_and_msc.py -p $cube_path -v tp --msc_only"
s_ssrd="python preprocessing/compute_ano_and_msc.py -p $cube_path -v ssrd --msc_only"
s_rh="python preprocessing/compute_ano_and_msc.py -p $cube_path -v rh_cf --msc_only"
s_lst="python preprocessing/compute_ano_and_msc.py -p $cube_path -v lst"
s_fvc="python preprocessing/compute_ano_and_msc.py -p $cube_path -v fvc"

dep="afterok:$JOBID1:$JOBID2"

JOBID3=$(sbatch -J dect2m --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $s_t2m)
JOBID4=$(sbatch -J dectp --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $s_tp)
JOBID5=$(sbatch -J decssrd --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $s_ssrd)
JOBID6=$(sbatch -J decrh --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $s_rh)
JOBID7=$(sbatch -J declst --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $s_lst)
JOBID8=$(sbatch -J decfvc --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $s_fvc)

# Calculating stats...

dep="afterok:$JOBID3:$JOBID4:$JOBID5:$JOBID6:$JOBID7:$JOBID8"

c1="python preprocessing/compute_stats.py -p $cube_path -n 20 -v canopyheight percent_tree_cover rootdepth sandfrac wtd"
c2="python preprocessing/compute_stats.py -p $cube_path -n 20 -v fvc fvc_ano fvc_msc"
c3="python preprocessing/compute_stats.py -p $cube_path -n 20 -v lst lst_ano lst_msc"
c4="python preprocessing/compute_stats.py -p $cube_path -n 20 -v rh_cf rh_cf_msc"
c5="python preprocessing/compute_stats.py -p $cube_path -n 20 -v ssrd ssrd_msc"
c6="python preprocessing/compute_stats.py -p $cube_path -n 20 -v t2m t2m_msc"
c7="python preprocessing/compute_stats.py -p $cube_path -n 20 -v tp tp_msc"

sbatch -J stageo --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c1
sbatch -J statfvc --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c2
sbatch -J statlst --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c3
sbatch -J statrh --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c4
sbatch -J statssrd --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c5
sbatch -J statt2m --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c6
sbatch -J stattp --dependency=$dep --parsable --cpus-per-task=20 preprocessing/slurm_python_call.sbatch $c7

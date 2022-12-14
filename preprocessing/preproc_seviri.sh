#!/bin/bash

# Data from 2004 to 2021

var="$1"
year="$2"

varu=$(echo $var | tr '[:lower:]' '[:upper:]')

PREPROC_DIR='/Net/Groups/BGI/scratch/bkraft/drought_data/preproc'
weights_file="$PREPROC_DIR/weights.${var}.${year}.w"

trap "rm -f $weights_file" EXIT
mkdir -p $PREPROC_DIR

remap_file () {
    file=$1

    if [ "$var" = "fvc" ]; then
        scale="daily"
    elif [ "$var" = "lst" ]; then
        scale="hourly"
    else
        printf "\n\033[0;31m%s\033[0m\n" "ERROR: variable '$var' is not valid, use one of ('lst' | 'fvc')."
        exit 1

    fi

    file_name=${file##*/}
    file_name=${file_name/LST/${var}.${scale}}
    file_out=${PREPROC_DIR}/$file_name
    min_file=${file_out/.nc/.min.nc}
    max_file=${file_out/.nc/.max.nc}

    cdo -O -s chname,LST,${var} -remap,preprocessing/griddes1d,$weights_file $file $file_out || exit 1

    #cdo -O -s -fldmin -timmin $file_out $min_file & \
    #cdo -O -s -fldmax -timmax $file_out $max_file || exit 1

    python preprocessing/add_minmax_attr.py \
        -p $file_out \
        -n $var \
        --clean_up || exit 1
}

cdo -O -s gennn,preprocessing/griddes1d \
    /Net/Groups/BGI/data/DataStructureMDI/DATA/grid/SEVIRI/0d10_daily/MD${varu}/Data/LST.1460.1140.01.${year}.nc \
    $weights_file || exit 1

install_pids=( )
if [ "$var" = "fvc" ]; then
    file_pattern="/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/SEVIRI/0d10_daily/MDFVC/Data/LST.1460.1140.*.${year}.nc"
elif [ "$var" = "lst" ]; then
    file_pattern="/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/SEVIRI/0d10_daily/MDLST/Data/LST_rm_gt_lt_2000/LST.1460.1140.*.${year}.nc"
fi

for file in $file_pattern
do
    ( remap_file $file || exit 1 ) & install_pids+=( $! )
done

while (( ${#install_pids[@]} )); do
  for pid_idx in "${!install_pids[@]}"; do
    pid=${install_pids[$pid_idx]}
    if ! kill -0 "$pid" 2>/dev/null; then # kill -0 checks for process existance
      # we know this pid has exited; retrieve its exit status
      wait "$pid" || exit
      unset "install_pids[$pid_idx]"
    fi
  done
  sleep 0.2 
done

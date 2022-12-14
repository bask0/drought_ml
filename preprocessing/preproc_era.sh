#!/bin/bash

# Data from 2002 to 2021

var="$1"
year="$2"

PREPROC_DIR="/Net/Groups/BGI/scratch/bkraft/drought_data/preproc"
weights_file="$PREPROC_DIR/weights.${var}.${year}.w"

trap "rm -f $weights_file" EXIT
mkdir -p $PREPROC_DIR

get_output_file () {
    # extracts varname, month, year from era file and creates nwe file
    # t2m.hh.an.era5_land.01.2005.nc -> t2m.hourly.1460.1140.01.2005.nc
    var="$(cut -d'.' -f1 <<<"$1")"
    mon="$(cut -d'.' -f5 <<<"$1")"
    year="$(cut -d'.' -f6 <<<"$1")"
    output_file=${var}.hourly.1460.1140.${mon}.${year}.nc
}

remap_file () {
    file=$1
    file_name=${file##*/}
    get_output_file $file_name
    file_out=${PREPROC_DIR}/$output_file
    min_file=${file_out/.nc/.min.nc}
    max_file=${file_out/.nc/.max.nc}

    cdo -O -s remap,preprocessing/griddes1d,$weights_file $file $file_out || exit 1

    #cdo -O -s -fldmin -timmin $file_out $min_file & \
    #cdo -O -s -fldmax -timmax $file_out $max_file || exit 1

    python preprocessing/add_minmax_attr.py \
        -p $file_out \
        -n $var \
        --clean_up || exit 1
}

cdo -O -s gennn,preprocessing/griddes1d \
    /Net/Groups/data_BGC/era5_land/e1/0d10_hourly/tp/${year}/tp.hh.fc.era5_land.01.${year}.nc \
    $weights_file || exit 1

install_pids=( )
for file in /Net/Groups/data_BGC/era5_land/e1/0d10_hourly/${var}/${year}/*.nc
do
    echo $file
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

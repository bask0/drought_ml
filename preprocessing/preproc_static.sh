#!/bin/bash

var=$1

PREPROC_DIR="/Net/Groups/BGI/scratch/bkraft/drought_data/preproc"

mkdir -p $PREPROC_DIR

remap_file () {
    file=$1
    varname=$2
    cdo -O -s --reduce_dim \
        -remapcon,/Net/Groups/BGI/people/bkraft/drought_ml/preprocessing/griddes1d \
        -sellonlatbox,-75,72,-38,77 $file \
        ${PREPROC_DIR}/${varname}.static.1460.1140.nc
}

if [ "$var" = "wtd" ]; then

    file=/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/WTD_Fan/v2013/Data/WTD.43200.21600.nc

elif [ "$var" = "mrd" ]; then

    file=/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d0083_static/MaxRootDepth/FAN_2017/Data/MaxRootDepth.FAN_2017.43200.000.21600.000.nc

elif [ "$var" = "twi" ]; then

    file=/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_static/topidx_ga/v1/Data/topidxga2.nc

elif [ "$var" = "sndppt" ]; then

    file=/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d10_static/soilgrids/v0_5_1/Data/SNDPPT.soilgrid.3600.1800.7.nc

elif [ "$var" = "ch" ]; then

    file=/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d083_static/GlobVeg3D/L3C/Data/GlobVeg3D.L3C.4320.2160.nc

elif [ "$var" = "tc" ]; then

    for year in 2003 2007 2011 2015 2019
    do
        file=/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d00416_annual/MODIS/MOD44B3.006/Data/Percent_Tree_Cover.86400.36000.${year}.nc 
        cdo -s -O -sellonlatbox,-75,72,-38,77 $file ${PREPROC_DIR}/tmp.tc.aoi.${year}.nc || exit 1
    done

    file=${PREPROC_DIR}/tmp.tc.median.nc

    cdo -O -s ensmedian ${PREPROC_DIR}/tmp.tc.aoi.*.nc $file || exit 1

    cdo -O --reduce_dim \
        -remapcon,/Net/Groups/BGI/people/bkraft/drought_ml/preprocessing/griddes1d \
        $file \
        ${PREPROC_DIR}/${var}.static.1460.1140.nc || exit 1

    # file=${PREPROC_DIR}/tmp.${var}.median.nc

    # cdo -O -s --reduce_dim \
    #     -remapcon,/Net/Groups/BGI/people/bkraft/drought_ml/preprocessing/griddes1d \
    #     -ensmedian ${PREPROC_DIR}/tmp.${var}.*.nc \
    #     ${PREPROC_DIR}/${var}.static.1460.1140.nc

    rm ${PREPROC_DIR}/tmp.*.nc

    exit 0

else

    printf "\n\033[0;31m%s\033[0m\n" "ERROR: variable '$var' is not valid, use one of ('wtd' | 'mrd' | 'twi' | 'sndppt' | 'ch' | 'tc')."
    exit 1

fi

remap_file $file $var || exit 1

exit 0

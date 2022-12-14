#!/bin/bash

NPROC=${1:-6}
RESUME=$2

timestamp=`date +"%Y.%m.%d.T%H.%M.%S"`
export LOG_DIR="/tmp/bkraft/dml/${timestamp}"
mkdir -p $LOG_DIR

DYNA_LOG_FILE="${LOG_DIR}/dynamic.log"
STAT_LOG_FILE="${LOG_DIR}/static.log"

proc_var () {
    var=$1
    year=$2

    if [ $year ]; then
        LOG_FILE="${LOG_DIR}/out_${var}_${year}.log"
    else
        LOG_FILE="${LOG_DIR}/out_${var}.log"
    fi

    if [[ "$var" =~ ^(fvc|lst)$ ]]; then
        if [ "$year" -ge 2004 ]; then
            bash preprocessing/preproc_seviri.sh $var $year > $LOG_FILE 2>&1
            st=$?
        fi
    elif [[ "$var" =~ ^(t2m|tp|ssrd|rH_cf)$ ]]; then
        bash preprocessing/preproc_era.sh $var $year > $LOG_FILE 2>&1
        st=$?
    elif [[ "$var" =~ ^(wtd|twi|mrd|sndppt|ch|tc)$ ]]; then
        bash preprocessing/preproc_static.sh $var > $LOG_FILE 2>&1
        st=$?
    else
        printf "\n\033[0;31m%s\033[0m\n" "> ERROR: variable '$var' is not valid."
        exit 1
    fi

    if [[ $st -gt 0 ]]
    then
        printf "\n\033[0;31m%s\033[0m\n" "> ERROR in subprocess (${LOG_FILE})"
        exit $st
    fi

} 

export -f proc_var

dt=$(date '+%d/%m/%Y %H:%M:%S')
SECONDS=0

printf "\e[1;34m%-6s\e[m\n" "[$dt] >>> start preprocessing data"

parallel --progress -j $NPROC --delay 5 --joblog $DYNA_LOG_FILE --lb $RESUME -- proc_var ::: lst fvc t2m tp ssrd rH_cf ::: {2002..2021}
parallel --progress -j $NPROC --delay 5 --joblog $STAT_LOG_FILE --lb $RESUME -- proc_var ::: wtd twi mrd sndppt ch tc

dt=$(date '+%d/%m/%Y %H:%M:%S')
duration=$SECONDS
elapsed="$(($duration / 60))m $(($duration % 60))s"

printf "\e[1;34m%-6s\e[m\n" "[$dt] >>> preprocessing done, $elapsed elapsed"

printf "\e[1;34m%-6s\e[m\n" "[$dt] >>> Dynamic variables processes:"
cat $DYNA_LOG_FILE 2>/dev/null
printf "\e[1;34m%-6s\e[m\n" "[$dt] >>> Static variables processes:"
cat $STAT_LOG_FILE 2>/dev/null

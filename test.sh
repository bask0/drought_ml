#!/bin/bash

Help()
{
# Display Help
echo "Run hyper parameter tuning script."
echo
echo "Syntax: tune.sh [OPTIONS] args"
echo "Options:"
echo "  -h    Print this Help."
echo "  -n    Maximum number of jobs to run in parallel, default is '1'."
echo "args:"
echo "  'full'    to fit the model and do predictions with best checkpoint (default)"
echo "  'fit'     to fit the model"
echo "  'predict' to do predictions with best checkpoint"

}

num_parallel=1
run_mode="full"

use_default=""

while getopts ":n:d" option; do
case $option in
    n)
    num_parallel=$OPTARG
    ;;
    d)
    use_default="-d"
    ;;
    :)
    printf "Error: missing argument for -%s\n" "$OPTARG" >&2
    exit 1
    ;;
    \?) # Invalid option
    echo "Error: Invalid option"
    exit
    ;;
esac
done

shift $((OPTIND-1))

if [ "$#" -eq 1 ]; then
    run_mode=$1
elif  [ "$#" -gt 1 ]; then
    echo "Error: either zero or one argument required, $(($#)) passed." >&2
    exit 1
fi

echo "sbatch --array 0-19%$num_parallel path/to/cmd.sh $run_mode $use_default"
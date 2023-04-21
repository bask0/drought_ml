
import inspect
from datetime import datetime


def get_tune_slurm_script(
        script_name: str,
        run_args: str,
        search_space_file: str,
        default_config_file: str) -> str:

    fit_cmd = f'srun python {script_name} fit -c {default_config_file} {run_args}'

    script = \
        """

        if [ "$#" -ne 0 ]; then
            printf "Error: tune script takes no arguments." >&2
            exit 1
        fi

        hps=$(awk -v ArrayTaskID=$(($SLURM_ARRAY_TASK_ID+1)) 'NR==ArrayTaskID{{ print; exit }}' {search_space_file})

        {fit_cmd} $hps
        """.format(fit_cmd=fit_cmd, search_space_file=search_space_file)

    return '\n' + inspect.cleandoc(script) + '\n'


def get_cv_slurm_script(
        script_name: str,
        run_args: str,
        search_space_file: str,
        version_dir: str,
        best_config_file: str,
        default_config_file: str,
        best_checkpoint_bath: str) -> str:

    fit_cmd = f'srun python {script_name} fit -c $config {run_args}'
    pred_cmd = f'srun python {script_name} predict -c $config {run_args} --ckpt_path {best_checkpoint_bath} --return_predictions false'

    script = \
        """

        use_best=true

        while getopts ":d" option; do
        case $option in
            d)
            use_best=false
            ;;
            \?) # Invalid option
            echo "Error: Invalid option"
            exit
            ;;
        esac
        done

        shift $((OPTIND-1))

        if [ "$#" -ne 1 ]; then
            echo "Error: CV script takes one argument, $(($#)) passed." >&2
            exit 1
        fi

        hps=$(awk -v ArrayTaskID=$(($SLURM_ARRAY_TASK_ID+1)) 'NR==ArrayTaskID{{ print; exit }}' {search_space_file})

        if [ "$use_best" = true ] ; then
            python ./project/utils/copy_best_checkpoint.py {version_dir} || exit $?
            config={best_config_file}
        else
            config={default_config_file}
        fi

        case "$1" in
            full)
            {fit_cmd} $hps
            wait
            {pred_cmd} $hps
            ;;
            fit)
            {fit_cmd} $hps
            ;;
            predict)
            {pred_cmd} $hps
            ;;
            *)
            echo "Error: Unknown option: [$1]. Valid options: [full | fit | predict]." >&2
            exit 1
            ;;
        esac
        """.format(
            fit_cmd=fit_cmd,
            pred_cmd=pred_cmd,
            best_config_file=best_config_file,
            default_config_file=default_config_file,
            version_dir=version_dir,
            search_space_file=search_space_file
        )

    return '\n' + inspect.cleandoc(script) + '\n'


def get_xai_slurm_script(
        version_dir: str,
        num_trials: int) -> str:

    script = \
        """

        num_trials={num_trials}

        num_split=0
        batch_size=20

        while getopts ":s:b:" option; do
        case $option in
            s)
            num_split=$OPTARG
            ;;
            b)
            batch_size=$OPTARG
            ;;
            \?) # Invalid option
            echo "Error: Invalid option"
            exit
            ;;
        esac
        done

        let "trial = $SLURM_ARRAY_TASK_ID / num_split"
        let "split_idx = $SLURM_ARRAY_TASK_ID % num_split"

        srun python integrated_gradients.py -p {version_dir} --gpu 0 --type cv --batch_size $batch_size --num_split $num_split --trial $trial --split_idx $split_idx
        """.format(
            version_dir=version_dir,
            num_trials=num_trials
        )

    return '\n' + inspect.cleandoc(script) + '\n'


def get_tune_run_script(script_path: str, num_trials: int) -> str:

    script = \
        """
        #!/bin/bash

        Help()
        {{
        # Display Help
        echo "Run hyper parameter tuning script."
        echo
        echo "Syntax: run_tune.sh [OPTIONS]"
        echo "Options:"
        echo "  -h    Print this Help."
        echo "  -n    Maximum number of jobs to run in parallel, default is '1'."

        }}

        num_parallel=1

        while getopts ":hn:" option; do
        case $option in
            h) # display Help
            Help
            exit
            ;;
            n)
            num_parallel=$OPTARG
            ;;
            :)
            printf "Error: missing argument for -%s\\n" "$OPTARG" >&2
            exit 1
            ;;
            \?) # Invalid option
            echo "Error: Invalid option" >&2
            exit
            ;;
        esac
        done

        shift $((OPTIND-1))

        if [ "$#" -gt 0 ]; then
            echo "Error: no arguments allowed, $(($#)) passed." >&2
            exit 1
        fi

        sbatch --wait --array 0-{num_trials}%$num_parallel {script_path}
        """.format(script_path=script_path, num_trials=num_trials - 1)

    return inspect.cleandoc(script) + '\n'


def get_cv_run_script(script_path: str, num_trials: int) -> str:

    script = \
        """
        #!/bin/bash

        Help()
        {{
        # Display Help
        echo "Run cross validation script."
        echo
        echo "Syntax: run_cv.sh [OPTIONS] args"
        echo "Options:"
        echo "  -h    Print this Help."
        echo "  -n    Maximum number of jobs to run in parallel, default is '1'."
        echo "  -d    Flag to enable cross validation with default configuration instead"
        echo "        of using the best parameters from hyper parameter tuning. If not passed,"
        echo "        hyper parameter tuning must be run first."
        echo "args:"
        echo "  'full'    to fit the model and do predictions with best checkpoint (default)"
        echo "  'fit'     to fit the model"
        echo "  'predict' to do predictions with best checkpoint"

        }}

        num_parallel=1
        run_mode="full"
        use_default=""

        while getopts ":hn:d" option; do
        case $option in
            h) # display Help
            Help
            exit
            ;;
            n)
            num_parallel=$OPTARG
            ;;
            d)
            use_default="-d"
            ;;
            :)
            printf "Error: missing argument for -%s\\n" "$OPTARG" >&2
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

        sbatch --wait --array 0-{num_trials}%$num_parallel {script_path} $run_mode $use_default
        """.format(script_path=script_path, num_trials=num_trials - 1)

    return inspect.cleandoc(script) + '\n'


def get_xai_run_script(script_path: str, num_folds: int) -> str:

    script = \
        """
        #!/bin/bash

        Help()
        {{
        # Display Help
        echo "Run xai script."
        echo
        echo "Syntax: tune.sh [OPTIONS] args"
        echo "Options:"
        echo "  -h    Print this Help."
        echo "  -s    Number of splits per dataloader, default is '1'."
        echo "  -b    Batch size, default is '10'."
        echo "  -n    Maximum number of jobs to run in parallel, default is '1'."

        }}

        num_folds={num_folds}

        num_parallel=1
        num_split=1
        batch_size=10

        while getopts ":hn:s:b:" option; do
        case $option in
            h) # display Help
            Help
            exit
            ;;
            n)
            num_parallel=$OPTARG
            ;;
            s)
            num_split=$OPTARG
            ;;
            b)
            batch_size=$OPTARG
            ;;
            :)
            printf "Error: missing argument for -%s\\n" "$OPTARG" >&2
            exit 1
            ;;
            \?) # Invalid option
            echo "Error: Invalid option"
            exit
            ;;
        esac
        done

        let "num_runs = num_folds * num_split - 1"

        sbatch --wait --array 0-$num_runs%$num_parallel {script_path} -s $num_split -b $batch_size
        """.format(script_path=script_path, num_folds=num_folds)

    return inspect.cleandoc(script) + '\n'


def get_full_run_script(tune_script_path: str, cv_script_path: str, xai_script_path: str) -> str:

    script = \
        """
        #!/bin/bash

        Help()
        {{
        # Display Help
        echo "Run hyper parameter tuning, cross validation, and xai script."
        echo
        echo "Syntax: tune.sh [OPTIONS]"
        echo "Options:"
        echo "  -h    Print this Help."
        echo "  -n    Maximum number of jobs to run in parallel, default is '1'."

        }}

        num_parallel=1

        while getopts ":hn:" option; do
        case $option in
            h) # display Help
            Help
            exit
            ;;
            n)
            num_parallel=$OPTARG
            ;;
            :)
            printf "Error: missing argument for -%s\\n" "$OPTARG" >&2
            exit 1
            ;;
            \?) # Invalid option
            echo "Error: Invalid option"
            exit
            ;;
        esac
        done

        shift $((OPTIND-1))

        if [ "$#" -gt 0 ]; then
            echo "Error: no arguments allowed, $(($#)) passed." >&2
            exit 1
        fi

        bash {tune_script_path} -n $num_parallel

        wait

        bash {cv_script_path} -n $num_parallel

        wait

        bash {xai_script_path} -n $num_parallel

        """.format(tune_script_path=tune_script_path, cv_script_path=cv_script_path, xai_script_path=xai_script_path)

    return inspect.cleandoc(script) + '\n'


def get_readme(
        call_cmd: str,
        full_script_path: str,
        tune_script_path: str,
        cv_script_path: str,
        xai_script_path: str) -> str:

    script = \
        """
        Automatically generated on {timestamp} with configuration:
        >>> {call_cmd}

        Run hyperparameter tuning ('tune'), cross-validation ('cv'), and integrated grads ('xai').

        tune, cv, and xai:
            bash {full_script_path} OPTIONS

            Options:
                -n NUM PARALLEL JOBS

            Examples:
                # Run hp tuning and cross validation with 4 parallel jobs.
                bash {full_script_path} -n 4

        tune:
            bash {tune_script_path} OPTIONS

            Options:
                -n NUM PARALLEL JOBS

            Examples:
                # Run hp tuning with 4 parallel jobs.
                bash {tune_script_path} -n 4

        cv:
            bash {cv_script_path} OPTIONS ARGS

            Options:
                -n NUM PARALLEL JOBS
                -d FLAG ENABLES CV WITH DEFAULT ARGS INSTEAD OF BEST HPS
            Arguments:
                full FIT AND PREDICT (DEFAULT)
                fit FIT ONLY
                predict PREDICT ONLY

            Examples:
                # Run cross validation (fit and predict) with 4 parallel jobs.
                bash {cv_script_path} -n 4
                # Run cross validation (only fit) with 4 parallel jobs (HPs from previous tuning).
                bash {cv_script_path} -n 4 fit
                # Run cross validation (only fit) with 4 parallel jobs (HPs from default config).
                bash {cv_script_path} -n 4 -d fit

        xai:
            bash {xai_script_path} OPTIONS

            Options:
                -n NUM PARALLEL JOBS
                -s NUM DATALOADER SPLIT
                -b BATCH SIZE

            Examples:
                # Run xai with 4 parallel jobs and a batch size of 30.
                bash {xai_script_path} -n 4 -b 30
                # Run xai with 20 parallel jobs and each fold split into two chunks.
                bash {xai_script_path} -n 20 -s 2

        """.format(
            call_cmd=call_cmd,
            timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            full_script_path=full_script_path,
            tune_script_path=tune_script_path,
            cv_script_path=cv_script_path,
            xai_script_path=xai_script_path)

    return inspect.cleandoc(script) + '\n'

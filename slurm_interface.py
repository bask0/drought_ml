
import os
import shutil
from argparse import ArgumentParser
from subprocess import call

from project.utils.slurm_utils import SlurmCluster
from project.utils.bash_script_templates import get_readme, get_full_run_script


if __name__ == '__main__':

    parser = ArgumentParser('SLURM interface')
    exp_args = parser.add_argument_group('experiment')

    exp_args.add_argument('--print_config', action='store_true')
    exp_args.add_argument('--job_name', type=str, default='default')
    exp_args.add_argument('--log_dir', type=str, default='./experiments')
    exp_args.add_argument('--search_space', type=str, required=False)
    exp_args.add_argument('--num_trials', type=int, default=24)

    cluster_args = parser.add_argument_group('cluster')
    cluster_args.add_argument('--per_experiment_nb_nodes', type=int, default=1)
    cluster_args.add_argument('--per_experiment_nb_cpus', type=int, default=32)
    cluster_args.add_argument('--per_experiment_nb_gpus', type=int, default=1)
    cluster_args.add_argument('--memory_mb_per_node', type=str, default='120G')
    cluster_args.add_argument('--gpu_type', type=str, default='A40')
    cluster_args.add_argument('--job_time', type=str, default='5-00:00:00')
    cluster_args.add_argument('--partition', type=str, default='gpu')

    args, unknownargs = parser.parse_known_args()

    cli_args = []
    for arg in unknownargs:
        if '=' in arg:
            cli_args.extend(arg.split('='))
        else:
            cli_args.append(arg)

    command = ' '.join(cli_args)

    # Print config to console.
    if args.print_config:
        call(
            f'python cli_interface.py fit {command} --print_config',
            shell=True,
            stdout=None
        )
        exit(0)

    cluster = SlurmCluster(
        script_name='cli_interface.py',
        root_dir=os.path.join(args.log_dir, args.job_name),
        job_name=args.job_name,
        modules=['cuda/11.6']
    )
    cluster.add_args(args)

    cluster.add_command('eval "$(conda shell.bash hook)"')
    cluster.add_command('source activate dml\n')

    cluster.add_command('export NCCL_DEBUG=ERROR')
    cluster.add_command('export PYTHONFAULTHANDLER=1')

    cluster.add_command('export LD_LIBRARY_PATH=/Net/Groups/BGI/scratch/bkraft/mamba/envs/dml/lib/:$LD_LIBRARY_PATH')

    tune_cmd_script_path = cluster.init_run(
        search_space_file=args.search_space,
        run_type='tune',
        num_trials=args.num_trials
    )

    cv_cmd_script_path = cluster.init_run(
        search_space_file=args.search_space,
        run_type='cv',
        num_trials=12  # num folds
    )

    # Print config to file (remove file if fails).
    r = call(
        f'python cli_interface.py fit {command} --print_config > {cluster.default_config_file}',
        shell=True,
        #stdout=None,
        stderr=None
    )

    if r > 0:
        shutil.rmtree(cluster.version_dir)
        exit(1)

    full_run_script = get_full_run_script(
        tune_script_path=tune_cmd_script_path,
        cv_script_path=cv_cmd_script_path
    )

    full_cmd_script_path = os.path.join(cluster.version_dir, 'run.sh')

    with open(full_cmd_script_path, mode='w') as file:
        file.write(full_run_script)

    readme = get_readme(
        full_script_path=full_cmd_script_path,
        tune_script_path=tune_cmd_script_path,
        cv_script_path=cv_cmd_script_path
    )

    slurm_cmd_script_path = os.path.join(cluster.version_dir, 'README.md')

    with open(slurm_cmd_script_path, mode='w') as file:
        file.write(readme)

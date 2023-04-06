

import os
from datetime import datetime
from argparse import Namespace
import numpy as np
import yaml
from yaml.loader import SafeLoader

from project.utils.bash_script_templates import \
    get_tune_slurm_script, get_cv_slurm_script, get_xai_slurm_script, get_tune_run_script, get_cv_run_script, get_xai_run_script


def sample_from_dict(d: dict):
    new_dict = {}

    for k0, v0 in d.items():
        for k1, v1 in v0.items():
            new_dict[f'{k0}.{k1}'] = np.random.choice(v1)

    return new_dict


class SlurmCluster(object):
    """Lightweight copy of test-tube's SlurmCluster by William Falcon
    https://github.com/williamFalcon/test-tube
    """
    def __init__(
            self,
            script_name: str,
            root_dir: str,
            job_name: str,
            enable_log_out: bool = True,
            log_dir: str | None = None,
            modules: list[str] | None = [],
            job_time: str = '01:00:00',
            minutes_to_checkpoint_before_walltime: int = 5,
            per_experiment_nb_gpus: int = 1,
            per_experiment_nb_cpus: int = 1,
            per_experiment_nb_nodes: int = 1,
            memory_mb_per_node: int = 2000,
            email: int = None,
            notify_on_end: bool = False,
            notify_on_fail: bool = False,
            partition: str | None = None,
            gpu_type: str | None = None,
            on_gpu: bool = False,
            commands: list[str] = [],
            slurm_commands: list[str] = [],
            hpc_exp_number: int = 0) -> None:
        self.root_dir = root_dir
        self.job_name = job_name

        self.enable_log_out = enable_log_out
        self.log_dir = log_dir
        self.modules = modules
        self.script_name = script_name
        self.job_time = job_time
        self.minutes_to_checkpoint_before_walltime = minutes_to_checkpoint_before_walltime
        self.per_experiment_nb_gpus = per_experiment_nb_gpus
        self.per_experiment_nb_cpus = per_experiment_nb_cpus
        self.per_experiment_nb_nodes = per_experiment_nb_nodes
        self.memory_mb_per_node = memory_mb_per_node
        self.email = email
        self.notify_on_end = notify_on_end
        self.notify_on_fail = notify_on_fail
        self.partition = partition
        self.gpu_type = gpu_type
        self.on_gpu = on_gpu
        self.commands = commands
        self.slurm_commands = slurm_commands
        self.hpc_exp_number = hpc_exp_number

    def add_args(self, args: Namespace):
        for key, value in args.__dict__.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_slurm_cmd(self, cmd, value, comment):
        self.slurm_commands.append((cmd, value, comment))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def _trialid_to_name(self, id: int) -> str:
        return f'trial{id:02d}'

    def _layout_logging_dir(self, num_trials: int, run_type: str, version: SyntaxWarning) -> None:
        """Generates dir structure for logging errors and outputs"""

        self.version_dir = os.path.join(self.root_dir, version)
        self.log_dir = os.path.join(self.root_dir, version, run_type)

        # if we have a test tube name, make the folder and set as the logging destination
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if run_type != 'xai':
            for i in range(num_trials):
                trial_dir = os.path.join(self.log_dir, self._trialid_to_name(i))
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)

        self.search_space_file = os.path.join(self.log_dir, 'search_space.txt')
        self.default_config_file = os.path.join(self.version_dir, 'default_config.yaml')

    def _build_slurm_command(
            self,
            run_type: str,
            num_trials: int,
            num_parallel: int,
            job_name: str | None = None,
            on_gpu: bool = True):

        hparams = {
            'trainer.logger.save_dir': self.log_dir,
            'trainer.logger.version': 'trial$(printf "%02d" $SLURM_ARRAY_TASK_ID)'
        }

        if run_type == 'cv':
            hparams.update({'trainer.max_epochs': 20})

        sub_commands = []

        command = [
            '#!/bin/bash',
            '#',
            '# Auto-generated',
            '#################\n'
        ]
        sub_commands.extend(command)

        # add job name
        command = [
            '# set a job name',
            '#SBATCH --job-name={}'.format(self.job_name if job_name is None else job_name),
            '#################\n',
        ]
        sub_commands.extend(command)

        # add out output
        if self.enable_log_out:
            if run_type == 'xai':
                out_path = os.path.join(self.log_dir, 'slurm_output_trial%2a_%j.out')
            else:
                out_path = os.path.join(self.log_dir, 'trial%2a', 'slurm_output_%j.out')
            command = [
                '# a file for job output, you can check job progress',
                '#SBATCH --output={}'.format(out_path),
                '#################\n',
            ]
            sub_commands.extend(command)

        # add job time
        command = [
            '# time needed for job',
            '#SBATCH --time={}'.format(self.job_time),
            '#################\n'
        ]
        sub_commands.extend(command)

        # add partition
        if self.partition is not None:
            command = [
                '# partition',
                '#SBATCH --partition={}'.format(self.partition),
                '#################\n'
            ]
            sub_commands.extend(command)

        # add nb of gpus
        if self.per_experiment_nb_gpus > 0 and on_gpu:
            command = [
                '# gpus per node',
                '#SBATCH --gres=gpu:{}'.format(self.per_experiment_nb_gpus),
                '#################\n'
            ]
            if self.gpu_type is not None:
                command = [
                    '# gpus per node',
                    '#SBATCH --gres=gpu:{}:{}'.format(self.gpu_type, self.per_experiment_nb_gpus),
                    '#################\n'
                ]
            sub_commands.extend(command)

        # add nb of cpus if not looking at a gpu job
        if self.per_experiment_nb_cpus > 0:
            command = [
                '# cpus per job',
                '#SBATCH --cpus-per-task={}'.format(self.per_experiment_nb_cpus),
                '#################\n'
            ]
            sub_commands.extend(command)

        # pick nb nodes
        command = [
            '# number of requested nodes',
            '#SBATCH --nodes={}'.format(self.per_experiment_nb_nodes),
            '#################\n'
        ]
        sub_commands.extend(command)

        # add job array specification
        command = [
            '# set job array',
            '#SBATCH --array=0-{}%{}'.format(num_trials - 1, num_parallel),
            '#################\n',
        ]
        sub_commands.extend(command)

        # pick memory per node
        command = [
            '# memory per node',
            '#SBATCH --mem={}'.format(self.memory_mb_per_node),
            '#################\n'
        ]
        sub_commands.extend(command)

        # add signal command to catch job termination
        command = [
            '# slurm will send a signal this far out before it kills the job',
            f'#SBATCH --signal=USR1@{self.minutes_to_checkpoint_before_walltime * 60}',
            '#################\n'
        ]
        sub_commands.extend(command)

        # Subscribe to email if requested
        mail_type = []
        if self.notify_on_end:
            mail_type.append('END')
        if self.notify_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0:
            mail_type_query = [
                '# Have SLURM send you an email when the job ends or fails',
                '#SBATCH --mail-type={}'.format(','.join(mail_type))
            ]
            sub_commands.extend(mail_type_query)

            email_query = [
                '#SBATCH --mail-user={}'.format(self.email),
            ]
            sub_commands.extend(email_query)

        # add custom sbatch commands
        sub_commands.append('\n')
        for (cmd, value, comment) in self.slurm_commands:
            comment = '# {}'.format(comment)
            cmd = '#SBATCH --{}={}'.format(cmd, value)
            spaces = '#################\n'
            sub_commands.extend([comment, cmd, spaces])

        # load modules
        sub_commands.append('\n')
        for module in self.modules:
            cmd = 'module load {}\n'.format(module)
            sub_commands.append(cmd)

        # remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # add additional commands
        sub_commands.append('\n')
        for cmd in self.commands:
            sub_commands.append(cmd)

        run_args = self._args_to_cmd(hparams)

        best_checkpoint_bath = os.path.join(
            self.log_dir, 'trial$(printf "%02d" $SLURM_ARRAY_TASK_ID)/checkpoints/best.ckpt')

        if run_type == 'tune':
            cmd = get_tune_slurm_script(
                script_name=self.script_name,
                run_args=run_args,
                search_space_file=self.search_space_file,
                default_config_file=self.default_config_file
            )
        elif run_type == 'cv':
            cmd = get_cv_slurm_script(
                script_name=self.script_name,
                run_args=run_args,
                search_space_file=self.search_space_file,
                version_dir=self.version_dir,
                best_config_file=os.path.join(self.log_dir, 'best_config.yaml'),
                default_config_file=self.default_config_file,
                best_checkpoint_bath=best_checkpoint_bath
            )
        elif run_type == 'xai':
            cmd = get_xai_slurm_script(
                version_dir=self.version_dir,
                num_trials=num_trials
            )

        sub_commands.append(cmd)

        # build full command with empty lines in between
        full_command = '\n'.join(sub_commands)
        return full_command

    def _should_escape(self, v):
        v = str(v)
        return '[' in v or ';' in v or ' ' in v

    def _args_to_cmd(self, hparams: Namespace | dict):
        params = []

        hparams_dict = hparams if isinstance(hparams, dict) else hparams.__dict__
        for k, v in hparams_dict.items():

            # don't add None params
            if v is None or v is False:
                continue

            # put everything in quotes except bools
            if self._should_escape(v):
                cmd = '--{} \"{}\"'.format(k, v)
            else:
                cmd = '--{} {}'.format(k, v)
            params.append(cmd)

        full_cmd = ' '.join(params)

        return full_cmd

    def _save_slurm_cmd(self, slurm_cmd: str, path: str = 'slurm_cmd.sh') -> str:
        slurm_cmd_script_path = os.path.join(self.log_dir, path)

        with open(slurm_cmd_script_path, mode='w') as file:
            file.write(slurm_cmd)

        return slurm_cmd_script_path

    def _save_run_cmd(self, run_cmd: str, run_type: str) -> str:
        run_cmd_script_path = os.path.join(self.version_dir, f'run_{run_type}.sh')

        with open(run_cmd_script_path, mode='w') as file:
            file.write(run_cmd)

        return run_cmd_script_path

    def init_run(
            self,
            search_space_file: str | None = None,
            run_type: str = 'tune',
            num_trials: int = 20,
            num_parallel: int = 1,
            exp_i: int = 0) -> tuple[str, str | None]:

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamp = f'exp_{exp_i:02d}_{timestamp}'

        self._layout_logging_dir(num_trials=num_trials, run_type=run_type, version=timestamp)

        if run_type == 'tune':
            if search_space_file is None:
                raise ValueError(
                    'must pass `search_space_file` as `run_type=\'tune\'`.'
                )
            with open(search_space_file, 'r') as f:
                hp_data = yaml.load(f, SafeLoader)

            samples = ''
            for _ in range(num_trials):
                hp_dict = sample_from_dict(hp_data)
                samples += ' '.join([f'--{k}={v}' for k, v in hp_dict.items()]) + '\n'
            samples = samples.strip()

            with open(self.search_space_file, 'w') as f:
                hp_data = f.write(samples)

        elif run_type == 'cv':
            with open(search_space_file, 'r') as f:
                hp_data = yaml.load(f, SafeLoader)

            samples = ''
            for i in range(num_trials):
                hp_dict = sample_from_dict(hp_data)
                samples += f'--data.fold_id={i}\n'
            samples = samples.strip()

            with open(self.search_space_file, 'w') as f:
                hp_data = f.write(samples)

        elif run_type == 'xai':
            pass

        else:
            raise ValueError(
                '`run_type` misconfiguration, must be \'tune\', \'cv\', or \'xai\', is \'{run_type}\'.'
            )

        slurm_cmd = self._build_slurm_command(
            run_type=run_type,
            num_trials=num_trials,
            num_parallel=num_parallel,
            job_name=run_type)

        tune_slurm_cmd_script_path = self._save_slurm_cmd(
            slurm_cmd=slurm_cmd
        )

        if run_type == 'tune':
            run_cmd = get_tune_run_script(script_path=tune_slurm_cmd_script_path, num_trials=num_trials)
        elif run_type == 'cv':
            run_cmd = get_cv_run_script(script_path=tune_slurm_cmd_script_path, num_trials=num_trials)
        else:
            run_cmd = get_xai_run_script(script_path=tune_slurm_cmd_script_path, num_folds=num_trials)

        run_cmd_script_path = self._save_run_cmd(run_cmd=run_cmd, run_type=run_type)

        return run_cmd_script_path

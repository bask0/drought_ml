
import os
from datetime import datetime
from argparse import ArgumentParser, Namespace
from subprocess import call


class SlurmCluster(object):
    """Lightweight copy of test-tube's SlurmCluster by William Falcon
    https://github.com/williamFalcon/test-tube
    """
    def __init__(
            self,
            script_file: str,
            log_path: str,
            job_name: str) -> None:
        self.log_path = log_path
        self.job_name = job_name

        self.enable_log_err = False
        self.enable_log_out = True
        self.err_log_path = None
        self.out_log_path = None
        self.modules = []
        self.script_name = script_file
        self.job_time = '01:00:00'
        self.minutes_to_checkpoint_before_walltime = 5
        self.per_experiment_nb_gpus = 1
        self.per_experiment_nb_cpus = 1
        self.per_experiment_nb_nodes = 1
        self.memory_mb_per_node = 2000
        self.email = None
        self.notify_on_end = False
        self.notify_on_fail = False
        self.job_name = None
        self.partition = None
        self.python_cmd = 'python3'
        self.gpu_type = None
        self.on_gpu = False
        self.commands = []
        self.slurm_commands = []
        self.hpc_exp_number = 0

    def add_slurm_cmd(self, cmd, value, comment):
        self.slurm_commands.append((cmd, value, comment))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def load_modules(self, modules):
        self.modules = modules

    def _layout_logging_dir(self, dryrun: bool = False) -> None:
        """Generates dir structure for logging errors and outputs"""

        # format the logging folder path
        slurm_out_path = os.path.join(self.log_path, self.job_name)

        self.log_path = slurm_out_path

        # if we have a test tube name, make the folder and set as the logging destination
        if not os.path.exists(slurm_out_path) and not dryrun:
            os.makedirs(slurm_out_path)

        # when err logging is enabled, build add the err logging folder
        if self.enable_log_err:
            err_path = os.path.join(slurm_out_path, 'slurm_err_logs')
            if not os.path.exists(err_path) and not dryrun:
                os.makedirs(err_path)
            self.err_log_path = err_path

        # when out logging is enabled, build add the out logging folder
        if self.enable_log_out:
            out_path = os.path.join(slurm_out_path, 'slurm_out_logs')
            if not os.path.exists(out_path) and not dryrun:
                os.makedirs(out_path)
            self.out_log_path = out_path

        # place where slurm files log to
        self.slurm_files_log_path = os.path.join(slurm_out_path, 'slurm_scripts')
        if not os.path.exists(self.slurm_files_log_path) and not dryrun:
            os.makedirs(self.slurm_files_log_path)

        # place where slurm files log to
        # self.pl_log_path = os.path.join(slurm_out_path, 'pl_logs')
        # if not os.path.exists(self.pl_log_path) and not dryrun:
        #     os.makedirs(self.pl_log_path)

    def _build_slurm_command(
            self,
            subcommand: str,
            hparams: Namespace | dict | None,
            config_files: list[str] | None,
            timestamp: str,
            exp_i: int = 0,
            on_gpu: bool = True):

        if hparams is None:
            hparams = {}

        hparams.update({
            #'trainer.default_root_dir': self.pl_log_path,
            'trainer.logger.save_dir': self.log_path,
            'trainer.logger.version': timestamp
        })

        sub_commands = []

        command =[
            '#!/bin/bash',
            '#',
            '# Auto-generated',
            '#################\n'
        ]
        sub_commands.extend(command)

        # add job name
        job_with_version = '{}_v{}'.format(self.job_name, exp_i)
        command = [
            '# set a job name',
            '#SBATCH --job-name={}'.format(job_with_version),
            '#################\n',
        ]
        sub_commands.extend(command)

        # add out output
        if self.enable_log_out:
            out_path = os.path.join(self.out_log_path, '{}_slurm_output_%j.out'.format(timestamp))
            command = [
                '# a file for job output, you can check job progress',
                '#SBATCH --output={}'.format(out_path),
                '#################\n',
            ]
            sub_commands.extend(command)

        # add err output
        if self.enable_log_err:
            err_path = os.path.join(self.err_log_path, '{}_slurm_output_%j.err'.format(timestamp))
            command = [
                '# a file for errors',
                '#SBATCH --error={}'.format(err_path),
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
        #sub_commands.append('\n')
        for module in self.modules:
            cmd = 'module load {}\n'.format(module)
            sub_commands.append(cmd)

        # remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # add additional commands
        for cmd in self.commands:
            sub_commands.append(cmd)

        if config_files is None:
            config_cmd = ''
        else:
            config_cmd = ''
            for config_file in config_files:
                if isinstance(config_file, list):
                    config_file = config_file[0]
                config_cmd += f'-c {config_file} '

        if hparams is None:
            run_args = ''
        else:
            run_args = self._args_to_cmd(hparams)

        raw_cmd = f'{self.python_cmd} {self.script_name} {subcommand} {config_cmd} {run_args}'
        cmd = f'\nsrun {raw_cmd}\n'
        sub_commands.append(cmd)

        # build full command with empty lines in between
        full_command = '\n'.join(sub_commands)
        return full_command, raw_cmd

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

    def _save_slurm_cmd(self, timestamp, slurm_cmd) -> str:
        slurm_cmd_script_path = os.path.join(self.slurm_files_log_path, '{}_slurm_cmd.sh'.format(timestamp))

        with open(slurm_cmd_script_path, mode='w') as file:
            file.write(slurm_cmd)

        return slurm_cmd_script_path

    def init_run(
            self,
            subcommand: str,
            hparams: Namespace | dict | None = None,
            config_files: list[str] | None = None,
            exp_i: int = 0,
            on_gpu: bool = True,
            dryrun: bool = False) -> tuple[str, str | None]:

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamp = 'trial_{}_{}'.format(exp_i, timestamp)

        self._layout_logging_dir(dryrun=dryrun)

        slurm_cmd, raw_cmd = self._build_slurm_command(
            subcommand=subcommand,
            hparams=hparams,
            config_files=config_files,
            timestamp=timestamp,
            exp_i=exp_i,
            on_gpu=on_gpu)

        if not dryrun:
            slurm_cmd_script_path = self._save_slurm_cmd(timestamp=timestamp, slurm_cmd=slurm_cmd)
        else:
            slurm_cmd_script_path = None

        return raw_cmd, slurm_cmd_script_path


if __name__ == '__main__':

    parser = ArgumentParser('SLURM interface')
    parser.add_argument('--check_args', action='store_true')
    parser.add_argument('--print_config', action='store_true')
    parser.add_argument('--job_name', type=str, default='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--config', '-c', action='append', nargs='+')

    args = parser.parse_args()

    cluster = SlurmCluster(
        script_file='project/clitest.py',
        log_path=args.log_dir,
        job_name=args.job_name
    )
    cluster.load_modules(['cuda/11.6'])

    cluster.per_experiment_nb_nodes = 1
    cluster.per_experiment_nb_cpus = 32
    cluster.per_experiment_nb_gpus = 1
    cluster.memory_mb_per_node = '250G'
    cluster.gpu_type = 'A40'
    cluster.job_time = '1-00:00:00'
    cluster.partition = 'gpu'
    cluster.job_name = 'test_job'

    # cluster.add_command('eval "$(conda shell.bash hook)"')
    # cluster.add_command('source activate dml\n')

    cluster.add_command('export NCCL_DEBUG=ERROR')
    cluster.add_command('export PYTHONFAULTHANDLER=1')

    hparams = {
    }

    raw_command, slurm_cmd_script_path = cluster.init_run(
        subcommand='fit',
        hparams=hparams,
        config_files=args.config,
        dryrun=args.check_args or args.print_config)

    if args.check_args:
        # os.system(raw_command + ' --help')
        call(raw_command + ' --help', shell=True)
    if args.print_config:
        # os.system(raw_command + ' --help')
        call(raw_command + ' --print_config', shell=True)
    else:
        call(f'sbatch {slurm_cmd_script_path}', shell=True)

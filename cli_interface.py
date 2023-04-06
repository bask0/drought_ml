
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from project.dataset import GeoDataQueue


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:

        # parser.link_arguments(
        #     'trainer.default_root_dir',
        #     'trainer.logger.init_args.save_dir',
        #     apply_on='parse')
        parser.link_arguments(
            'data.num_features_hourly',
            'model.init_args.num_inputs',
            apply_on='instantiate')
        parser.link_arguments(
            'data.num_features_static',
            'model.init_args.num_geofactors',
            apply_on='instantiate')
        parser.add_argument(
            '--search_space', type=str, required=False
        )


def pl_cli(CLI: LightningCLI, run: bool = True) -> LightningCLI:
    default_config_file = {'default_config_files': ['config/base.yaml']}

    if run:
        trainer_defaults = {
            subcommand: default_config_file for subcommand in ['fit', 'validate', 'test', 'predict', 'tune']
        }
    else:
        trainer_defaults = default_config_file

    cli = CLI(
        datamodule_class=GeoDataQueue,
        parser_kwargs=trainer_defaults,
        run=run,
        save_config_kwargs={'overwrite': True}
    )

    return cli


if __name__ == '__main__':

    cli = pl_cli(CustomLightningCLI, run=True)

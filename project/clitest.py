
from project.pl_models.tcn_model import TemporalConvNetPL
from dataset import GeoDataQueue

from pytorch_lightning.cli import LightningCLI

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.num_features_hourly', 'model.init_args.num_inputs', apply_on='instantiate')
        parser.link_arguments('data.num_features_static', 'model.init_args.num_geofactors', apply_on='instantiate')
        parser.link_arguments('data.num_targets_daily', 'model.init_args.num_outputs', apply_on='instantiate')

def pl_cli(run: bool = True):
    default_config_file = {'default_config_files': ['project/config/base.yaml']}
    trainer_defaults = {
        subcommand: default_config_file for subcommand in ['fit', 'validate', 'test', 'predict', 'tune']
    }
    if run:
        cli = CustomLightningCLI(datamodule_class=GeoDataQueue, parser_kwargs=trainer_defaults, run=run)

        cli.trainer.predict(
            dataloaders=cli.datamodule,
            return_predictions=False,
            ckpt_path='best'
        )

    if hasattr(cli.trainer, 'overfit_batches'):
        if run and cli.trainer.overfit_batches > 0:
            cli.trainer.predict(
                dataloaders=cli.datamodule.train_dataloader(),
                return_predictions=False,
                ckpt_path='best'
            )

    return cli


if __name__ == '__main__':
    pl_cli()

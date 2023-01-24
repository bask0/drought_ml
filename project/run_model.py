

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from test_tube.hpc import SlurmCluster
from project.pl_models.tcn_model import TemporalConvNetPL
from dataset import GeoDataQueue
from utils.pl_utils import OutputWriter
from argparse import ArgumentParser, Namespace
import os

cluster = SlurmCluster(
    hyperparam_optimizer=Namespace(a=1, b=2),
    log_path="/some/path/to/save",
)
cluster.optimize_parallel_slurm

JOBID = os.getenv('SLURM_JOB_ID')
# from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler

#import logging; logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# profiler = AdvancedProfiler(dirpath='.', filename='perf_logs')
# profiler = PyTorchProfiler(dirpath='.', filename='perf_logs', group_by_input_shapes=True)

# TODO:
# - [x] Save predictions
# - [x] NLL loss
# - [x] CLI setup
# - [x] SLURM integration
# - [ ] Switch data / hour dimensions
# - [ ] Predict anomalies instead of raw
# - [ ] Set up fast dev run
# - [ ] XVAL
# - [ ] Prediction setup (also think about saving predictions)

def main(overfit: bool = False):
    chunk_buffer_size = 1 if overfit else 4
    num_queue_workers = 1 if overfit else 12
    accumulate_grad_batches = 1 if overfit else 5
    overfit_batches = 5 if overfit else 0.0
    limit_train_batches = 0.0 if overfit else 0.05
    limit_val_batches = 0.05 if overfit else 0.0
    limit_predict_batches = overfit_batches if overfit else 0.0
    max_epochs = 100 if overfit else 20

    geodata = GeoDataQueue(
        fold_id=0,
        target_daily=['fvc_ano'],
        batch_size=5,
        chunk_buffer_size=chunk_buffer_size,
        num_queue_workers=num_queue_workers,
    )

    tcn = TemporalConvNetPL(
        num_inputs=geodata.num_features_hourly,
        num_geofactors=geodata.num_features_static,
        num_outputs=1,
        num_hidden=128,
        num_layers=2,
        dropout=0.1,
        static_dropout=0.0,
        kernel_size=3,
        lr=0.00001,
        weight_decay=0.0
    )

    output_writer = OutputWriter(targets=geodata.targets)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_predict_batches=limit_predict_batches,
        callbacks=[output_writer],
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        plugins=[SLURMEnvironment(auto_requeue=False)],
        #logger=TensorBoardLogger(version=JOBID, name='logs')
        # profiler=profiler,
        # fast_dev_run=2,
    )

    trainer.fit(tcn, datamodule=geodata)

    if overfit:
        trainer.predict(tcn, dataloaders=geodata.train_dataloader(), return_predictions=False)
    else:
        trainer.predict(tcn, datamodule=geodata, return_predictions=False, ckpt_path='best')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--overfit', action='store_true')

    args = argparser.parse_args()

    main(overfit=args.overfit)

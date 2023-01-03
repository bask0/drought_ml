

import pytorch_lightning as pl
from project.pl_models.tcn_model import TemporalConvNetPL
from dataset import GeoDataQueue

# from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler

# import logging; logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# profiler = AdvancedProfiler(dirpath='.', filename='perf_logs')
# profiler = PyTorchProfiler(dirpath='.', filename='perf_logs', group_by_input_shapes=True)

# TODO:
# - NLL loss
# - Save rpedictions
# - Predict anomalies instead of raw
# - SLURM integration
# - Set up fast dev run

geodata = GeoDataQueue(
    fold_id=0,
    batch_size=4,
    chunk_buffer_size=4,
    num_queue_workers=12,
)

tcn = TemporalConvNetPL(
    num_inputs=geodata.num_features_hourly,
    num_geofactors=geodata.num_features_static,
    num_outputs=1,
    num_hidden=128,
    num_layers=3,
    dropout=0.2,
    static_dropout=0.2,
    kernel_size=5,
    tasks='fvc',
    lr=0.0001,
    weight_decay=0.0
)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=[7],
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    accumulate_grad_batches=5,
    # overfit_batches=5,
    # profiler=profiler,
    # fast_dev_run=2,
)
trainer.fit(tcn, datamodule=geodata)

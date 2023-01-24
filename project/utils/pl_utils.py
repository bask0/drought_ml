import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import xarray as xr
import os
import shutil
import numpy as np
import torch.multiprocessing as mp
from torch import Tensor
from collections import namedtuple
from typing import Any, Sequence

from dataset import QueueFiller
from utils.types import ReturnPattern


class OutputWriter(BasePredictionWriter):
    """Output writer to use as callback in prediction mode."""
    def __init__(
            self,
            overwrite: bool | str = True) -> None:
        """Initialize OutputWriter.

        Results are written to the trainer logging directory as `predictions.zarr`. Predictions
        are expected in `ReturnPattern` format. Note that `ReturnPattern.var_hat` (the optional
        uncertainties) may be `None`. In this case, no variable is created for uncertainties and
        only the mean is written to file. For each target, the resulting zarr file has the variables
        - '<target>_hat' (mean)
        - '<target>_vhat' (optional variance)

        Example:
        >>> pred_writer = OutputWriter()
        >>> trainer = Trainer(callbacks=[pred_writer])
        >>> model = MyModel()
        >>> trainer.predict(model, return_predictions=False)

        Args:
            overwrite: if `True` (default), existing predictions are overridden. Else, an error
                is raised if the target exists.
        """
        super().__init__(write_interval='batch')

        self.overwrite = overwrite

        self.targets: list[str] = None
        self.zarr_file: xr.Dataset | None = None
        self.mask: xr.DataArray | None = None
        self.chunk_coords: np.typing.ArrayLike | None = None
        self.num_year_samples: int = None

        self.processes: list[mp.Process] = []
        self.chunks = {}

    def write_on_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            prediction: ReturnPattern,
            batch_indices: Sequence[int] | None,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int) -> None:

        if self.zarr_file is None:
            self.has_variance = prediction.var_hat is not None
            chunk_size = self.init_zarr(
                trainer=trainer, dataloader_idx=dataloader_idx, has_variance=self.has_variance)
            self.chunk_bounds_lat = QueueFiller.coords2bounds(self.mask, dim='lat', chunk_size=chunk_size)
            self.chunk_bounds_lon = QueueFiller.coords2bounds(self.mask, dim='lon', chunk_size=chunk_size)

        for i in range(len(prediction.coords.chunk)):
            i_pred = self.subset_namedtuple(prediction, i)
            chunk_id = i_pred.coords.chunk

            if chunk_id not in self.chunks:
                chunk_lat, chunk_lon = self.chunk_coords[chunk_id]

                lat_bounds = slice(*self.chunk_bounds_lat[chunk_lat])
                lon_bounds = slice(*self.chunk_bounds_lon[chunk_lon])
                num_samples = self.mask.isel(lat=lat_bounds, lon=lon_bounds).sum().compute()

                chunk_ds = self.get_target_like(
                    self.zarr_file.isel(lat=lat_bounds, lon=lon_bounds),
                    has_variance=self.has_variance,
                    use_orig_name=False
                )

                self.chunks[chunk_id] = {
                    'ds': chunk_ds,
                    'num_samples': num_samples * self.num_year_samples,
                    'num_saved': 0,
                    'lat_bounds': lat_bounds,
                    'lon_bounds': lon_bounds
                }

            for target_i, target in enumerate(self.targets):
                self.chunks[chunk_id]['ds'][
                    self.get_target_pred_name(name=target, is_variance=False)
                ].loc[{
                    'lat': i_pred.coords.lat,
                    'lon': i_pred.coords.lon,
                    'time': slice(i_pred.coords.window_start, i_pred.coords.window_end)}] =\
                            i_pred.mean_hat[-i_pred.coords.num_days:, ..., target_i]

                if self.has_variance:
                    self.chunks[chunk_id]['ds'][
                        self.get_target_pred_name(name=target, is_variance=True)
                    ].loc[{
                        'lat': i_pred.coords.lat,
                        'lon': i_pred.coords.lon,
                        'time': i_pred.coords.time_slice}] =\
                            i_pred.var_hat[-i_pred.coords.num_days:, ..., target_i]

            self.chunks[chunk_id]['num_saved'] += 1

            if self.chunks[chunk_id]['num_saved'] == self.chunks[chunk_id]['num_samples']:
                process = mp.Process(target=self.write_chunk, args=(self.chunks[chunk_id],))
                self.processes.append(process)
                process.start()
                del self.chunks[chunk_id]

            remove_processes = []
            for process_i, process in enumerate(self.processes):
                if not process.is_alive():
                    remove_processes.append(process_i)

            if len(remove_processes) > 0:

                for process_i in reversed(remove_processes):
                    process = self.processes.pop(process_i)
                    process.terminate()
                    process.join()

    def on_predict_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence[Any]) -> None:

        for chunk in self.chunks.values():
            self.write_chunk(chunk)

        self.chunks = {}

    def write_chunk(self, chunk: dict) -> None:
        chunk_ds = chunk['ds'].drop_vars(['time'])

        chunk_ds.to_zarr(self.zarr_file.encoding['source'], consolidated=True, region={
            'lat': chunk['lat_bounds'],
            'lon': chunk['lon_bounds']
        })

    def init_zarr(
            self, trainer: 'pl.Trainer',
            dataloader_idx: int,
            has_variance: bool) -> tuple[str, xr.Dataset, xr.DataArray, np.typing.ArrayLike, int]:

        zarr_dir = os.path.join(trainer.log_dir, 'predictions.zarr')

        if os.path.isdir(zarr_dir):
            if self.overwrite:
                shutil.rmtree(zarr_dir)

            else:
                raise FileExistsError(
                    f'The zar file {zarr_dir} already exists. Set `OutputWriter(..., overwrite=True)` '
                    'to overwrite the file.'
                )

        dataset = trainer.predict_dataloaders[dataloader_idx].dataset
        data = dataset.data

        missing_targets = []
        for target in self.targets:
            if not target in data.data_vars:
                missing_targets.append(target)

        if len(missing_targets) > 0:
            raise KeyError(
                f'some target(s) missing in dataset: {missing_targets}.'
            )

        dummy = self.get_target_like(data, has_variance=has_variance, use_orig_name=True)
        dummy.to_zarr(zarr_dir, compute=False)

        self.targets = dataset.targets
        self.zarr_file = xr.open_zarr(zarr_dir)
        self.mask = dataset.mask
        self.chunk_coords = dataset.chunk_coords
        self.num_year_samples = dataset.num_year_samples

        return dataset.chunk_size

    def get_target_like(self, data: xr.Dataset, has_variance: bool, use_orig_name: bool) -> xr.Dataset:
        dummy = xr.Dataset()
        for target in self.targets:
            name = target if use_orig_name else self.get_target_pred_name(name=target, is_variance=False)
            dummy[self.get_target_pred_name(name=target, is_variance=False)] = data[name]
            if has_variance:
                dummy[self.get_target_pred_name(name=target, is_variance=True)] = data[name]

        return xr.full_like(dummy, fill_value=np.nan)

    @staticmethod
    def get_target_pred_name(name: str, is_variance: bool) -> bool:
        if is_variance:
            return name + '_vhat'
        else:
            return name + '_hat'

    @staticmethod
    def isnamedtuple(obj) -> bool:
        return (
                isinstance(obj, tuple) and
                hasattr(obj, '_asdict') and
                hasattr(obj, '_fields')
        )

    def subset_namedtuple(self, x: Any, ind: int) -> namedtuple:
        el_type = type(x)

        el_list = []
        for el in x:
            if self.isnamedtuple(el):
                el_subset = self.subset_namedtuple(el, ind)
            elif el is None:
                el_subset = None
            else:
                el_subset = el[ind]
                if isinstance(el_subset, str):
                    # For dates in string format.
                    pass
                elif el_subset.ndim == 0:
                    # For numpy floats and integers.
                    el_subset = el_subset.item()
                elif isinstance(el_subset, Tensor):
                    # For Tensors.
                    el_subset = el_subset.cpu().numpy()
                else:
                    raise TypeError(
                        '`el_subset` is neither a string, a float, an integer, nor a Tensor.'
                    )

            el_list.append(el_subset)

        return el_type(*el_list)

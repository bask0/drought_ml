import torch
from torch.utils.data import IterableDataset, DataLoader, default_collate
import pytorch_lightning as pl


import xarray as xr
import dask
import numpy as np
from collections import namedtuple
import torch.multiprocessing as mp
import queue
import time

import logging
from typing import Any, Iterable

BatchPattern = namedtuple('BatchPattern', 'f_hourly f_static t_hourly, t_daily coords')
Coords = namedtuple('Coords', 'lat, lon')

dask.config.set(scheduler='synchronous')


class NotEnoughSamplesError(Exception):
    """Exception raised if not enough samples are left to fill a batch.
    """

    def __init__(self, message='Not enough samples left.'):
        super().__init__(message)


class GracefulExit(object):
    """Context manager for graceful shutdown of subprocesses and queues."""
    def __init__(self, processes: list[mp.Process], manager: mp.Manager):
        self.processes = processes
        self.manager = manager

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for process in self.processes:
            process.terminate()
            process.join()
        self.manager.shutdown()


def collate_handle_none(x: Any) -> Any:

    if x[0] is None:
        return None
    else:
        return default_collate(x)


def batchpattern_collate(batch: list[BatchPattern]) -> BatchPattern:
    """Custom collate function to handle None values in BatchPattern.

    Note that this function is not intended to handle sparse None values, but the case
    where None is returned for one or more elements in BatchPattern, i.e., completely empty\
    features or targets.

    Args:
        batch: a list of BatchPattern.

    Returns:
        A BatchPattern corresponding to stacked individual elements of BatchPatterns.

    """
    elem_type = type(batch[0])

    return elem_type(*(collate_handle_none(x=samples) for samples in zip(*batch)))


class DataChunk(object):
    """Defines a single chunk and how to retreive data from it."""
    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            targets_hourly: list[str] = [],
            targets_daily: list[str] = [],
            feature_scaling: dict[str, dict[str, float]] = None,
            dtype: str = 'float32',
            disable_shuffling: bool = False,
            dummy_data: bool = False) -> None:

        if dummy_data:
            data = xr.zeros_like(data).load() #xr.zeros_like(data.isel(time=slice(0, 10))).load()
        else:
            data = data.load()

        mask = mask.stack(sample=('lat', 'lon')).reset_coords(drop=True)
        data = data.stack(sample=('lat', 'lon')).reset_coords(drop=True)
        self.data = data.where(mask, drop=True).load()

        self.features_hourly = features_hourly
        self.features_static = features_static
        self.targets_hourly = targets_hourly
        self.targets_daily = targets_daily

        self.feature_scaling = feature_scaling
        self.dtype = dtype
        self.disable_shuffling = disable_shuffling
        self.dummy_data = dummy_data

        self.coords = self._get_coords()

        self._current_sample = 0

    def next(self) -> BatchPattern:
        sample = self.coords[self._current_sample]
        self._current_sample += 1

        data_sel = self.data.isel(sample=sample)

        return BatchPattern(
            f_hourly=self.xr2numpy(
                data_sel[self.features_hourly], scale=True),
            f_static=self.xr2numpy(
                data_sel[self.features_static], scale=True),
            t_hourly=self.xr2numpy(data_sel[self.targets_hourly], scale=False
            ),
            t_daily=self.xr2numpy(data_sel[self.targets_daily], scale=False
            ),
            coords=Coords(
                lat=data_sel.lat.item(),
                lon=data_sel.lon.item())
        )

    def get_n_next(self, n: int):
        result = []
        for _ in range(n):
            result.append(self.next())

        return result

    def xr2numpy(self, x: xr.Dataset, scale: bool) -> np.ndarray | None:

        if len(x) == 0:
            return None

        if scale:
            if self.feature_scaling is None:
                raise ValueError(
                    'no `feature_scaling` has been passed but scling was requested.'
                )
            for var in x.data_vars:
                if var not in self.feature_scaling:
                    raise KeyError(
                        f'scling for variable `{var}` requested, but not present in `feature_scaling`.'
                    )
                scaling = self.feature_scaling[var]
                data_min = scaling['min']
                data_max = scaling['max']
                x[var] = (x[var] - data_min) * 2 / (data_max - data_min) - 1

        return x.to_array('var').transpose('hour', 'time', 'var', missing_dims='ignore').values.astype(self.dtype)

    def _get_coords(self) -> Iterable:
        coords = np.arange(len(self.data.sample))
        if not self.disable_shuffling:
            np.random.shuffle(coords)
        return coords

    @property
    def num_left(self) -> int:
        return len(self.coords) - self._current_sample


class QueueFiller(object):
    """Defines a collection of buffered chunks and how to iterate them."""

    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            chunk_mask: xr.DataArray,
            batch_queue: mp.Queue,
            index_queue: mp.Queue,
            batch_size: int,
            chunk_buffer_size: int,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            targets_hourly: list[str] = [],
            targets_daily: list[str] = [],
            drop_last: bool = False,
            chunk_size: int = 20,
            disable_shuffling: bool = False,
            dummy_data: bool = False) -> None:
        """Initialize QueueFiller.

        Args:
            ds: the dataset containing spatio-temporal observations. Must at least have
                the dimensions `lat` and `lon`.
            mask: a spatio-temporal mask indicating valid pixels (`True`). Must at least have
                the dimensions `lat` and `lon` equal to `ds`.
            chunk_mask: a spatio-temporal mask indicating valid chunks (`True`). Must at least have
                the dimensions `lat` and `lon`.
            batch_queue: a multiprocessing.Manager().Queue for queuing of batches.
            index_queue: a multiprocessing.Manager().Queue holding the chunk indices.
            batch_size: the batch size, an integer larger than 0.
            chunk_buffer_size: the size of the queue, i.e., max number of batches to preload.
            features_hourly: a list of hourly dynamic features, must be data variables of `ds`.
            features_static: a list of static features, must be data variables of `ds`.
            targets_hourly: a list of hourly dynamic targets, must be data variables of `ds`.
            targets_daily: a list of daily dynamic targets, must be data variables of `ds`.
            drop_last: Whether to drop the last batch with size < `batch_size`. Default is `False`.
            chunk_size: the latitude/longitude chunk sizes. Must divide the
                `ds.lat`/`ds.lon` chunk size without remainder.
            disable_shuffling: if `True` shuffling will be turned off. Default is `False`.
            dummy_data: if set to `True`, dummy data is returned and reading from disk is omitted; use for debugging.
                Default is `False`.

        """

        super().__init__()

        self.data = data
        self.mask = mask.load()
        self.chunk_mask = chunk_mask.load()
        self.batch_queue = batch_queue
        self.index_queue = index_queue

        self.features_hourly = features_hourly
        self.features_static = features_static
        self.targets_hourly = targets_hourly
        self.targets_daily = targets_daily

        self.batch_size = batch_size
        self.chunk_buffer_size = chunk_buffer_size
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.disable_shuffling = disable_shuffling
        self.dummy_data = dummy_data

        self.feature_scaling = self._get_scaling(self.data, self.features_hourly, self.features_static)
        self.chunk_coords = np.argwhere(self.chunk_mask.values)
        self.chunk_bounds_lat = self.coords2bounds('lat')
        self.chunk_bounds_lon = self.coords2bounds('lon')

        self.chunks: list[DataChunk] = []

    def get_n_next(self, n: int, return_remaining: bool = False):
        sizes = self._distribute_batch(n=n, return_all=return_remaining)
        res = [c.get_n_next(s) for c, s in zip(self.chunks, sizes)]
        res = [item for sublist in res for item in sublist]
        self.clean_empty_chunks()

        return res

    def fill(self) -> None:
        torch.set_num_threads(1)

        has_more_chunks = True

        while True:
            try:
                if not has_more_chunks:
                    if (self.num_left == 0) or ((self.num_left < self.batch_size) and self.drop_last):
                        self.chunks = []
                        break

                if has_more_chunks:

                    if self.num_buffered < self.chunk_buffer_size or self.num_left < self.batch_size:
                        try:
                            index = self.index_queue.get_nowait()
                            chunk_lat, chunk_lon = self.chunk_coords[index]
                            lat_bounds = slice(*self.chunk_bounds_lat[chunk_lat])
                            lon_bounds = slice(*self.chunk_bounds_lon[chunk_lon])

                            data_chunk = self.data.isel(lat=lat_bounds, lon=lon_bounds)
                            mask_chunk = self.mask.isel(lat=lat_bounds, lon=lon_bounds)

                            self.chunks.append(
                                DataChunk(
                                    data=data_chunk,
                                    mask=mask_chunk,
                                    features_hourly=self.features_hourly,
                                    features_static=self.features_static,
                                    targets_daily=self.targets_daily,
                                    targets_hourly=self.targets_hourly,
                                    feature_scaling=self.feature_scaling,
                                    disable_shuffling=self.disable_shuffling,
                                    dummy_data=self.dummy_data
                                )
                            )
                            continue

                        except queue.Empty:
                            has_more_chunks = False
                            continue

                batch = self.get_n_next(min(self.num_left, self.batch_size))
                batch = batchpattern_collate(batch)

                self.batch_queue.put(batch)

            except Exception as e:
                self.batch_queue.put(e)

    def clean_empty_chunks(self) -> None:
        empty = []
        for i, chunk in enumerate(self.chunks):
            if chunk.num_left == 0:
                empty.append(i)

        for i in reversed(empty):
            self.chunks.pop(i)

    def coords2bounds(self, dim: str):
        return np.lib.stride_tricks.sliding_window_view(range(0, len(self.mask[dim]) + 1, self.chunk_size), 2)

    def _distribute_batch(self, n: int, return_all: bool = False) -> list[int]:
        num_left = self.num_left_per_chunk

        if return_all:
            return num_left

        if sum(num_left) < n:
            raise NotEnoughSamplesError()

        suggested_sizes = [min(num_left[i], int(n // len(num_left))) for i in range(len(num_left))]
        num_missing = n - sum(suggested_sizes)

        i = 0
        while num_missing > 0:
            i_ = i % len(suggested_sizes)
            if num_left[i_] - suggested_sizes[i_] > 0:
                suggested_sizes[i_] += 1
                num_missing -= 1
            i += 1

        return suggested_sizes

    def _get_scaling(
            self,
            data: xr.Dataset,
            features_hourly: list[str],
            features_static: list[str]) -> dict[str, dict[str, float]]:

        min_max = {}

        for var in features_hourly + features_static:
            data_min = data[var].attrs['data_min']
            data_max = data[var].attrs['data_max']
            min_max.update({var: {'min': data_min, 'max': data_max}})

        return min_max

    def coords2bounds(self, dim: str):
        return np.lib.stride_tricks.sliding_window_view(range(0, len(self.mask[dim]) + 1, self.chunk_size), 2)

    @property
    def num_buffered(self) -> int:
        return len(self.chunks)

    @property
    def num_left(self) -> int:
        return sum(chunk.num_left for chunk in self.chunks)

    @property
    def num_left_per_chunk(self) -> int:
        return [chunk.num_left for chunk in self.chunks]

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_coords)

    @property
    def num_features_hourly(self) -> int:
        return len(self.features_hourly)

    @property
    def num_features_static(self) -> int:
        return len(self.features_static)

    @property
    def num_targets_hourly(self) -> int:
        return len(self.targets_hourly)

    @property
    def num_targets_daily(self) -> int:
        return len(self.targets_daily)


class DataQueue(IterableDataset):
    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            batch_size: int,
            queue_size: int,
            chunk_buffer_size: int,
            num_queue_workers: int = 12,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            targets_hourly: list[str] = [],
            targets_daily: list[str] = [],
            drop_last: bool = False,
            chunk_size: int = 20,
            disable_shuffling: bool = False,
            dummy_data: bool = False):
        super().__init__()

        self.data = data
        self.mask = mask.load()
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.chunk_buffer_size = chunk_buffer_size
        self.num_queue_workers = num_queue_workers
        self.features_hourly = features_hourly
        self.features_static = features_static
        self.targets_hourly = targets_hourly
        self.targets_daily = targets_daily
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.disable_shuffling = disable_shuffling
        self.dummy_data = dummy_data

        self.num_samples = self.mask.sum().compute().item()
        self.chunk_mask = (self.mask.coarsen(lat=chunk_size, lon=chunk_size).sum() > 0).compute()

        self._check_ds(self.data, self.mask)

    def get_queue_filler(
            self,
            batch_queue: mp.Queue,
            index_queue: mp.Queue) -> QueueFiller:

        queue_filler = QueueFiller(
            data=self.data,
            mask=self.mask,
            chunk_mask=self.chunk_mask,
            batch_queue=batch_queue,
            index_queue=index_queue,
            batch_size=self.batch_size,
            chunk_buffer_size=self.chunk_buffer_size,
            features_hourly=self.features_hourly,
            features_static=self.features_static,
            targets_daily=self.targets_daily,
            targets_hourly=self.targets_hourly,
            drop_last=False,
            disable_shuffling=self.disable_shuffling,
            dummy_data=self.dummy_data
        )

        return queue_filler

    def init_queue_fillers(
            self,
            batch_queue: mp.Queue,
            index_queue: mp.Queue,
            num_processes: int = 1) -> tuple[list[mp.Process], int]:

        processes = []
        for _ in range(num_processes):
            qf = self.get_queue_filler(batch_queue=batch_queue, index_queue=index_queue)

            process = mp.Process(target=qf.fill, daemon=True)
            processes.append(process)

        return processes, qf.num_chunks

    def __iter__(self):

        manager = mp.Manager()
        batch_queue = manager.Queue(maxsize=self.queue_size)
        index_queue = manager.Queue()

        processes, num_chunks = self.init_queue_fillers(
            batch_queue=batch_queue,
            index_queue=index_queue,
            num_processes=self.num_queue_workers
        )

        with GracefulExit(processes=processes, manager=manager):

            if self.disable_shuffling:
                chunk_ids = np.random.permutation(num_chunks)
            else:
                chunk_ids = np.arange(num_chunks)

            for i in chunk_ids:
                index_queue.put(i)

            for process in processes:
                process.start()

            subsize_batches = []

            has_more_batches = True

            while has_more_batches:
                try:
                    el = batch_queue.get_nowait()

                    if isinstance(el, Exception):
                        raise el

                    if len(el.coords.lat) < self.batch_size:
                        subsize_batches.append(el)
                    else:
                        yield el

                except queue.Empty:
                    if all([not process.is_alive() for process in processes]):
                        if not index_queue.empty():
                            raise AssertionError(
                                'all queue-filler process finished before index queue was emptied.'
                            )

                        has_more_batches = False
                    else:
                        time.sleep(0.1)

            batch_items = []
            for incomplete_batch in subsize_batches:
                for i in range(len(incomplete_batch.coords.lat)):
                    batch_item = BatchPattern(
                        *[el if el is None else el[i] for el in incomplete_batch[:4]],
                        coords=Coords(incomplete_batch.coords.lat[i], incomplete_batch.coords.lon[i]))
                    batch_items.append(batch_item)
                    if len(batch_items) == self.batch_size:
                        batch = batchpattern_collate(batch_items)
                        batch_items = []
                        yield batch

            if not self.drop_last and len(batch_items) > 0:
                batch = batchpattern_collate(batch_items)
                yield batch


    def get_dummy_batch(self):
        return BatchPattern(
            f_hourly=torch.randn(self.batch_size, 24, 1000, len(self.features_hourly)) if self.features_hourly else None,
            f_static=torch.randn(self.batch_size, len(self.features_static)) if self.features_static else None,
            t_daily=torch.randn(self.batch_size, 1000, len(self.targets_daily)) if self.targets_daily else None,
            t_hourly=torch.randn(self.batch_size, 24, 1000, len(self.targets_hourly)) if self.targets_hourly else None,
            coords=Coords(lat=torch.arange(self.batch_size), lon=torch.arange(self.batch_size))
        )

    def _check_ds(
            self,
            ds: xr.Dataset,
            mask: xr.DataArray) -> None:
        """Checks dataset 'ds' properties.

        - `ds` must have chunks
        - `ds` and `mask` must have `lat` and `lon` dimensions
        - chunks must be defined for `lat` and `lon`

        Each negative check throws and error.
        """
        ch = ds.chunks
        if not len(ch):
            raise ValueError(
                'no chunks defined for the dataset `ds`.'
            )

        missing_dim = ''
        for d, d_name in zip([ds, mask], ['ds', 'mask']):
            for dim in ['lat', 'lon']:
                if dim not in d.dims:
                    missing_dim += f' `{d_name}.{dim}`'
        if missing_dim != '':
            raise ValueError(
                'the inputs `ds` and `mask` both most have dimensions `lat` and `lon`, but the '
                f'following dimensions are not present: {missing_dim}'
            )

    def __len__(self) -> int:

        import math

        if self.drop_last:
            length = self.num_samples // self.batch_size
        else:
            length = math.ceil(self.num_samples / self.batch_size)

        return length


class GeoDataQueue(pl.LightningDataModule):
    """Pytorch lightning data module for SEVIRI geo-spcatial data.

    Notes
    -----
    The class defines dataloaders for geo-spatial data. Spatial chunks (contiguous on disk) are queued
    and sampling is done from the loaded chunks. This speeds up data-loading by minimizing the number
    of read operatinos. Access dataloaders as:
     `GeoDataQueue(...)[train_dataloader | val_dataloader | test_dataloader | predict_dataloader]`
    The latter two return the same test data.

    Create a new `GeoDataQueue` for each crossvalidaiton split. Splitting is done spatially using
    the spatial regions of interests (ROIs) as defined in the data cube `fold_mask` variable. One fold is used
    for validation and testing, the remaining ones are used for training.

    The dataloaders use `num_workers=0` as parallelization is done with `n=num_queue_workers` processes by
    running a queue generator in the background.

    Data shape
    ----------
    The dataloaders return data in the format:
    > BatchPattern = namedtuple('BatchPattern', 'f_hourly f_static t_hourly, t_daily coords'),
    The `coords` are formatted as:
    > Coords = namedtuple('Coords', 'lat, lon')

    `f_hourly` are hourly features. They have shape [B, H, D, FH]
    `f_static` are static features. They have shape [B, FS]
    `t_hourly` are hourly targets. They have shape [B, H, D, TH]
    `f_daily` are daily targets. They have shape [B, D, TD]
    `lat`, `lon` are latitudes and longitudes of each samples, they both have shape [B]

    * batch_size: B
    * num_hours: H (=24)
    * num_days: D
    * num_features_hourly: FH
    * num_features_static: FS
    * num_targets_hourly: TH
    * num_targets_daily: TD

    Access data in a batch as:
    `batch = next(iter(dataloader))`
    `batch.f_hourly`
    `batch.coords.lat`
    etc.

    All data is returned as type `float32`. Features are standardized from their original range to
    the range [-1, 1] based on the precomputed statistics from the FULL DATASET. This may lead to
    information leaking, but it is too computationally to compute data statistics on the fly.

    """
    def __init__(
            self,
            fold_id: int,
            num_folds: int = 6,
            batch_size: int = 50,
            chunk_buffer_size: int = 4,
            queue_size: int = 100,
            num_queue_workers: int = 12,
            chunk_size: int = 20,
            disable_shuffling: bool = False,
            cube_path: str = '/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr',
            dummy_data: bool = False,
            **dataloader_kwargs):
        """GeoDataQueue initialization.

        Args:
            fold_id: the fold ID, an integer in the range (0, `num_folds` - 1).
            num_folds: the number of folds, a positive integer that matches the number of folds
                in the `fold_mask` (a variable of the dataset). Default is 6.
            batch_size: the batch size, an integer > 0. Default is 50.
            chunk_buffer_size: the number of spatial chunks to be buffered, an integer > 0.
                Note that smaller values lead to less randomness in the samples (they tend to come from
                the same spatial chunk for continuous batches), and large values lead to long initialization
                time. Values between 3 and 6 are suggested. Default is 4.
            queue_size: the number of batches to queue up.
            num_queue_workers: number of workers that fill batch queue in parallel.
            chunk_size: the latitude/longitude chunk sizes. Must divide the
                `ds.lat`/`ds.lon` chunk size without remainder.
            disable_shuffling: if `True` shuffling will be turned off for all dataloaders. By default,
                shuffling is turned on for all dataloaders. If Pytorch Lightning's overfit
                features (`pl.Trainer(overfit_batches=n)`) is used, `disable_shuffling=True` is be forced.
            cube_path: the path to the data cube (zarr format).
                Default is '/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr'.
            dummy_data: if set to `True`, dummy data is returned and reading from disk is omitted; use for debugging.
                Default is `False`.
            dataloader_kwargs: keyword arguments passed to `torch.Dataset(...)`
        """
        super().__init__()

        self.fold_id = fold_id
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.chunk_buffer_size = chunk_buffer_size
        self.queue_size = queue_size
        self.num_queue_workers = num_queue_workers
        self.chunk_size = chunk_size
        self.disable_shuffling = disable_shuffling
        self.dummy_data = dummy_data
        self.dataloader_kwargs = dataloader_kwargs

        self.cube_path = cube_path
        self.ds = xr.open_zarr(self.cube_path)

        self.train_folds, self.valid_folds, self.test_folds = \
            self.get_fold_split(fold_id=self.fold_id, num_folds=self.num_folds)

        self.fold_mask = self.ds.fold_mask.load()

        self.features_hourly = [
            'rh_cf', 'ssrd', 't2m', 'tp'
        ]
        self.features_static = [
            'canopyheight', 'rootdepth', 'percent_tree_cover', 'sandfrac', 'topidx', 'wtd'
        ]
        self.targets_daily = [
            'fvc'
        ]
        self.targets_hourly = [
            #'lst'
        ]

    @staticmethod
    def get_fold_split(fold_id: int, num_folds: int) -> tuple[list[int], list[int], list[int]]:

        if (fold_id < 0) or (fold_id > num_folds - 1):
            raise ValueError(
                f'`fold_id` out of range; must be an integer in the range (0, {num_folds - 1})'
            )

        folds = {fold for fold in range(num_folds)}
        val_fold = fold_id
        test_fold = (fold_id + 3) % 6
        folds = folds - {val_fold, test_fold}
        folds = list(folds)

        return folds, [val_fold], [test_fold]

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_mask = self.fold_mask.sel(fold=self.train_folds).any('fold').load()
            self.valid_mask = self.fold_mask.sel(fold=self.valid_folds).any('fold').load()
        elif stage == 'validate':
            self.valid_mask = self.fold_mask.sel(fold=self.valid_folds).any('fold').load()
        elif stage == 'test':
            self.test_mask = self.fold_mask.sel(fold=self.test_folds).any('fold').load()
        elif stage == 'predict':
            self.predict_mask = self.fold_mask.sel(fold=self.test_folds).any('fold').load()
        else:
            raise ValueError(
                f'`stage`  must be one of \'fit\', \'validate\', \'test\', or \'predict\',  is \'{stage}\'.'
            )

    def teardown(self, stage: str | None = None) -> None:
        if stage == 'fit':
            del self.train_mask
            del self.valid_mask
        elif stage == 'validate':
            del self.valid_mask
        elif stage == 'test':
            del self.test_mask
        elif stage == 'predict':
            del self.predict_mask
            raise ValueError(
                f'`stage`  must be one of \'fit\', \'validate\', \'test\', or \'predict\',  is \'{stage}\'.'
            )

    def get_dataloader(self, cvset: str):
        if cvset == 'train':
            mask = self.train_mask
            chunk_buffer_size = self.chunk_buffer_size
        elif cvset == 'valid':
            mask = self.valid_mask
            chunk_buffer_size = 1
        elif cvset == 'test':
            mask = self.test_mask
            chunk_buffer_size = 1
        elif cvset == 'predict':
            mask = self.predict_mask
            chunk_buffer_size = 1
        else:
            raise ValueError(
                f'`cvset`  must be one of \'train\', \'valid\', \'test\', or \'predict\',  is \'{cvset}\'.'
            )

        if self.trainer.overfit_batches > 0:
            disable_shuffling = True
        else:
            disable_shuffling = self.disable_shuffling

        dataqueue = DataQueue(
            data=self.ds,
            mask=mask,
            batch_size=self.batch_size,
            num_queue_workers=self.num_queue_workers,
            queue_size=self.queue_size,
            chunk_buffer_size=chunk_buffer_size,
            features_hourly=self.features_hourly,
            features_static=self.features_static,
            targets_daily=self.targets_daily,
            targets_hourly=self.targets_hourly,
            drop_last=False,
            disable_shuffling=disable_shuffling,
            dummy_data=self.dummy_data
        )

        dataloader = DataLoader(
            dataqueue,
            num_workers=0,  # Cannot be > 0 as we spawn subprocesses in dataqueue.
            batch_size=None,
            shuffle=False,  # Do not change, has no impact as we use a custom IterableDataset.
            **self.dataloader_kwargs)

        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader('train')
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader('valid')
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader('test')
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader('predict')
        return dataloader

    @property
    def num_features_hourly(self) -> int:
        return len(self.features_hourly)

    @property
    def num_features_static(self) -> int:
        return len(self.features_static)

    @property
    def num_targets_daily(self) -> int:
        return len(self.targets_daily)

    @property
    def num_targets_hourly(self) -> int:
        return len(self.targets_hourly)

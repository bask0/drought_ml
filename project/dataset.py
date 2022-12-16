import torch
from torch.utils.data import IterableDataset, DataLoader, default_collate
import pytorch_lightning as pl

import xarray as xr
import dask
import numpy as np
from collections import namedtuple
import multiprocessing as mp
import time

from typing import Any


BatchPattern = namedtuple('BatchPattern', 'f_hourly f_static t_hourly, t_daily coords')
Coords = namedtuple('Coords', 'lat, lon')

dask.config.set(scheduler='synchronous')


class NotEnoughSamplesError(Exception):
    """Exception raised if not enough samples are left to fill a batch.
    """

    def __init__(self, message='Not enough samples left.'):
        super().__init__(message)


class NoMoreChunksError(Exception):
    """Exception raised if no chunks are left.
    """

    def __init__(self, message='No chunks left.'):
        super().__init__(message)


def worker_init_fn(worker_id: int):
    "Desynchonize workers."
    if worker_id > 0:
        time.sleep(worker_id * 3)


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
    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            shuffle: bool = False,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            targets_hourly: list[str] = [],
            targets_daily: list[str] = [],
            feature_scaling: dict[str, dict[str, float]] = None,
            dtype: str = 'float32') -> None:

        self.data = data.load()
        self.mask = mask

        self.features_hourly = features_hourly
        self.features_static = features_static
        self.targets_hourly = targets_hourly
        self.targets_daily = targets_daily

        self.feature_scaling = feature_scaling
        self.dtype = dtype

        self.coords = np.argwhere(mask.values)
        if shuffle:
            np.random.shuffle(self.coords)

        self._current_sample = 0

    def next(self) -> BatchPattern:
        lat, lon = self.coords[self._current_sample]
        self._current_sample += 1

        data_sel = self.data.isel(lat=lat, lon=lon)

        return BatchPattern(
            f_hourly=self.xr2numpy(data_sel[self.features_hourly], scale=True),
            f_static=self.xr2numpy(data_sel[self.features_static], scale=True),
            t_hourly=self.xr2numpy(data_sel[self.targets_hourly], scale=False),
            t_daily=self.xr2numpy(data_sel[self.targets_daily], scale=False),
            coords=Coords(lat=data_sel.lat.item(), lon=data_sel.lon.item())
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

    @property
    def num_left(self) -> int:
        return len(self.coords) - self._current_sample


class Buffer(object):
    def __init__(self) -> None:
        self.chunks = []

    def add_chunk(self, chunk) -> None:
        self.chunks.append(chunk)

    def clean_empty_chunks(self) -> None:
        empty = []
        for i, chunk in enumerate(self.chunks):
            if chunk.num_left == 0:
                empty.append(i)

        for i in reversed(empty):
            self.chunks.pop(i)

    def get_n_next(self, n: int, return_remaining: bool = False):
        sizes = self._distribute_batch(n=n, return_all=return_remaining)
        res = [c.get_n_next(s) for c, s in zip(self.chunks, sizes)]
        res = [item for sublist in res for item in sublist]
        self.clean_empty_chunks()
        return res

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_left(self) -> int:
        return sum(chunk.num_left for chunk in self.chunks)

    @property
    def num_left_per_chunk(self) -> int:
        return [chunk.num_left for chunk in self.chunks]

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


class BufferedDataset(IterableDataset):
    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            batch_size: int,
            num_buffer: int,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            targets_hourly: list[str] = [],
            targets_daily: list[str] = [],
            drop_last: bool = False,
            chunk_size: int = 20) -> None:
        """Initialize BufferedDataset.

        Args:
            ds: the dataset containing spatio-temporal observations. Must at least have
                the dimensions `lat` and `lon`.
            mask: a spatio-temporal mask indicating valid pixels (`True`). Must at least have
                the dimensions `lat` and `lon` equal to `ds`.
            sample_chunk_size: the smallest sample unit. E.g., with `sample_chunk_size = 10`, a block
                of size 10 x 10 is read at once and returned.
            features_hourly: a list of hourly dynamic features, must be data variables of `ds`.
            features_static: a list of static features, must be data variables of `ds`.
            targets_hourly: a list of hourly dynamic targets, must be data variables of `ds`.
            targets_daily: a list of daily dynamic targets, must be data variables of `ds`.
            drop_last: Whether to drop the last batch with size < batch_size. Default is `False`.
            chunk_size: the latitude/longitude chunk sizes. Must divide the
                `ds.lat`/`ds.lon` chunk size without remainder.

        """

        super().__init__()

        self.data = data
        self.mask = mask.load()
        self.features_hourly = [features_hourly] if isinstance(features_hourly, str) else features_hourly
        self.features_static = [features_static] if isinstance(features_static, str) else features_static
        self.targets_hourly = [targets_hourly] if isinstance(targets_hourly, str) else targets_hourly
        self.targets_daily = [targets_daily] if isinstance(targets_daily, str) else targets_daily
        self._check_ds(ds=self.data, mask=self.mask)
        self.feature_scaling = self._get_scaling(self.data, self.features_hourly, self.features_static)

        self.num_samples = self.mask.sum().compute().item()
        self.chunk_mask = mask.coarsen(lat=chunk_size, lon=chunk_size).sum().compute()
        self.batch_size = batch_size
        self.num_buffer = num_buffer
        self.drop_last = drop_last
        self.chunk_size = chunk_size

        self.chunk_coords = np.argwhere(self.chunk_mask.values)

        self.chunk_bounds_lat = self.coords2bounds('lat')
        self.chunk_bounds_lon = self.coords2bounds('lon')

        self._was_shuffled = mp.Value('i', 0)
        self._current_chunk = mp.Value('i', 0)
        self._shared_indices = mp.Array('i', self.num_chunks)
        self._shared_indices[:] = [i for i in range(self.num_chunks)]

    def _get_scaling(
            self,
            data: xr.Dataset,
            features_hourly: list[str],
            features_static: list[str]) -> dict[str, dict[str, float]]:

        min_max = {}

        # for var in features_hourly:
        #     enc = data[var].encoding
        #     data_min = enc['add_offset'] - 30000 * enc['scale_factor']
        #     data_max = enc['scale_factor'] * 60000 + data_min
        #     min_max.update({var: {'min': data_min, 'max': data_max}})

        for var in features_hourly + features_static:
            data_min = data[var].attrs['data_min']
            data_max = data[var].attrs['data_max']
            min_max.update({var: {'min': data_min, 'max': data_max}})

        return min_max

    def coords2bounds(self, dim: str):
        return np.lib.stride_tricks.sliding_window_view(range(0, len(self.mask[dim]) + 1, self.chunk_size), 2)


    def next_chunk(self) -> DataChunk:

        with self._current_chunk.get_lock():
            current_chunk = self._current_chunk.value
            if current_chunk >= self.num_chunks:
                raise NoMoreChunksError()

            current_chunk_nr = self._shared_indices[current_chunk]
            chunk_lat, chunk_lon = self.chunk_coords[current_chunk_nr]
            self._current_chunk.value += 1

        lat_bounds = slice(*self.chunk_bounds_lat[chunk_lat])
        lon_bounds = slice(*self.chunk_bounds_lon[chunk_lon])
        data_chunk = self.data.isel(lat=lat_bounds, lon=lon_bounds)
        mask_chunk = self.mask.isel(lat=lat_bounds, lon=lon_bounds)
        return DataChunk(
            data=data_chunk,
            mask=mask_chunk,
            shuffle=True,
            features_hourly=self.features_hourly,
            features_static=self.features_static,
            targets_daily=self.targets_daily,
            targets_hourly=self.targets_hourly,
            feature_scaling=self.feature_scaling,
        )

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

    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            if self.drop_last:
                length = int(self.num_samples // self.batch_size)
            else:
                length = int(np.ceil(self.num_samples // self.batch_size))
        else:
            if self.drop_last:
                length = int(self.num_samples / worker_info.num_workers // self.batch_size)
            else:
                length = int(self.num_samples / worker_info.num_workers // self.batch_size) + worker_info.num_workers

        return length

    def __iter__(self) -> BatchPattern:

        with self._was_shuffled.get_lock():
            if self._was_shuffled.value == 0:
                self._shared_indices[:] = np.random.permutation(self._shared_indices)
                self._current_chunk.value = 0
                self._was_shuffled.value = 1

        buffer = Buffer()
        is_last_chunk = False
        is_last_batch = False
        while True:
            # No more batches
            if is_last_batch:
                self._was_shuffled.value = 0
                return

            # Add chunk and start over or set last chunk.
            if not is_last_chunk:
                try:
                    if buffer.num_chunks < self.num_buffer:
                        next_chunk = self.next_chunk()
                        buffer.add_chunk(next_chunk)

                        continue

                except NoMoreChunksError as e:
                    is_last_chunk = True

            # Get next batch.
            try:
                # If enough samples in buffer, return them.
                yield batchpattern_collate(buffer.get_n_next(self.batch_size))
            except NotEnoughSamplesError as e:
                # If not enough samples in buffer...
                # ...try to add chunk if more available, and start over.
                try:
                    next_chunk = self.next_chunk()
                    buffer.add_chunk(next_chunk)
                    continue
                except NoMoreChunksError as e:
                    # ...or get last batch if not drop_last and any data left.
                    is_last_batch = True
                    if (not self.drop_last) and (buffer.num_left > 0):
                        yield batchpattern_collate(buffer.get_n_next(self.batch_size, return_remaining=True))


class BufferedGeoDataLoader(pl.LightningDataModule):
    """Pytorch lightning data module for SEVIRI geo-spcatial data.

    Notes
    -----
    The class defines dataloaders for geo-spatial data. Spatial chunks (contiguous on disk) are buffered
    and sampling is done from the loaded chunks. This speeds up data-loading by minimizing the number
    of read operatinos. Access dataloaders as:
     `BufferedGeoDataLoader(...)[train_dataloader | val_dataloader | test_dataloader | predict_dataloader]`
    The latter two return the same test data.

    Create a new `BufferedGeoDataLoader` for each crossvalidaiton split. Splitting is done spatially using
    the spatial regions of interests (ROIs) as defined in the data cube `fold_mask` variable. One fold is used
    for validation and testing, the remaining ones are used for training.

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
            num_chunk_preload: int = 4,
            num_wokers: int = 0,
            cube_path: str = '/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr',
            **dataset_kwargs):
        """GeoDataLoader initialization.

        Parameters
        ----------
        fold_id: the fold ID, an integer in the range (0, `num_folds` - 1).
        num_folds: the number of folds, a positive integer that matches the number of folds
            in the `fold_mask` (a variable of the dataset). Default is 6.
        batch_size: the batch size, an integer > 0. Default is 50.
        num_chunk_preload: the number of spatial chunks to be buffered per worker, an integer > 0.
            Note that smaller values lead to less randomness in the samples (they tend to come from
            the same spatial chunk for continuous batches), and large values lead to long initialization
            time. Values between 3 and 6 are suggested. Default is 4.
        num_workers: number of parallel subprocesses to use for data loading. 0 means that the data
            will be loaded in the main process. Default is 0.
        cube_path: the path to the data cube (zarr format).
            Default is '/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr'.
        dataset_kwargs: keyword arguments passed to `torch.Dataset(...)`
        """
        super().__init__()

        self.fold_id = fold_id
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_chunk_preload = num_chunk_preload
        self.num_workers = num_wokers
        self.dataset_kwargs = dataset_kwargs

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
            'fvc_ano'
        ]
        self.targets_hourly = [
            #'lst'
        ]

    def get_dummy_batch(self):
        return BatchPattern(
            f_hourly=torch.randn(self.batch_size, 24, 1000, len(self.features_hourly)) if self.features_hourly else None,
            f_static=torch.randn(self.batch_size, len(self.features_static)) if self.features_static else None,
            t_daily=torch.randn(self.batch_size, 1000, len(self.targets_daily)) if self.targets_daily else None,
            t_hourly=torch.randn(self.batch_size, 24, 1000, len(self.targets_hourly)) if self.targets_hourly else None,
            coords=Coords(lat=torch.arange(self.batch_size), lon=torch.arange(self.batch_size))
        )

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

    def get_dataloader(self, mask: xr.DataArray) -> DataLoader:
        buffered_dataset = BufferedDataset(
            data=self.ds,
            mask=mask,
            batch_size=self.batch_size,
            num_buffer=self.num_chunk_preload,
            features_hourly=self.features_hourly,
            features_static=self.features_static,
            targets_daily=self.targets_daily,
            targets_hourly=self.targets_hourly,
            drop_last=False
        )

        dataloader = DataLoader(
            buffered_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            **self.dataset_kwargs
        )

        return dataloader

    def train_dataloader(self) -> DataLoader:
        fold_mask = self.fold_mask.sel(fold=self.train_folds).any('fold').load()
        dataloader = self.get_dataloader(mask=fold_mask)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        fold_mask = self.fold_mask.sel(fold=self.valid_folds).any('fold').load()
        dataloader = self.get_dataloader(mask=fold_mask)
        return dataloader

    def test_dataloader(self) -> DataLoader:
        fold_mask = self.fold_mask.sel(fold=self.test_folds).any('fold').load()
        dataloader = self.get_dataloader(mask=fold_mask)
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        fold_mask = self.fold_mask.sel(fold=self.test_folds).any('fold').load()
        dataloader = self.get_dataloader(mask=fold_mask)
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

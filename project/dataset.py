import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader, default_collate
import pytorch_lightning as pl
import pandas as pd
import xarray as xr
import dask
import numpy as np
from scipy.ndimage import binary_dilation
import torch.multiprocessing as mp
import queue
import time
import traceback
from itertools import product
import warnings
from typing import Any, Iterable
from numpy.typing import ArrayLike

from project.utils.geo import msc_align
from project.utils.types import DateLookup, Coords, BatchPattern, QueueException

# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*does not have many workers.*')
warnings.filterwarnings('ignore', '.*Your `IterableDataset` has `__len__` defined.*')

# We *must* use synchronous scheduler to avoid deadlock.
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
    where None is returned for one or more elements in BatchPattern, i.e., completely empty
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
            target_hourly: list[str] = [],
            target_daily: list[str] = [],
            context_size: int = 2,
            window_size: int = -1,
            chunk_index: int = 0,
            data_scaling: dict[str, dict[str, float]] = None,
            unfold_msc: bool = True,
            dtype: str = 'float32',
            disable_shuffling: bool = False,
            dummy_data: bool = False,
            load_data: bool = True,
            return_baseline: bool = False,
            precip_zero_baseline: bool = False) -> None:

        if dummy_data:
            data = xr.zeros_like(data)
        else:
            data = data

        mask = mask.stack(sample=('lat', 'lon')).reset_coords(drop=True)
        data = data.stack(sample=('lat', 'lon')).reset_coords(drop=True)

        if load_data:
            self.data = data.where(mask, drop=True).load()
        else:
            self.data = data.where(mask, drop=True)

        self.features_hourly = features_hourly
        self.features_hourly_msc = [f + '_msc' for f in features_hourly]
        self.features_static = features_static
        self.target_hourly = target_hourly
        self.target_daily = target_daily

        self.context_size = context_size
        self.window_size = window_size
        self.chunk_index = chunk_index

        if data_scaling is None:
            raise ValueError('Data scaling must be provied.')
        self.data_scaling = data_scaling
        self.unfold_msc = unfold_msc
        self.dtype = dtype
        self.disable_shuffling = disable_shuffling
        self.dummy_data = dummy_data
        self.return_baseline = return_baseline
        self.precip_zero_baseline = precip_zero_baseline

        self.coords = self._get_coords()

        self._current_sample = 0

    def next(self) -> BatchPattern:

        sample, time = self.coords[self._current_sample]
        self._current_sample += 1

        data_sel = self.data.isel(sample=sample).sel(time=time.time_slice)

        dayofyear = data_sel.time.dt.dayofyear - 1

        if self.unfold_msc:
            for var in self.target_daily + self.target_hourly:
                data_sel[var + '_msc'] = data_sel[var + '_msc'].sel(dayofyear=dayofyear)

        data_sel_hourly = data_sel[self.features_hourly]
        data_sel_stat = data_sel[self.features_static]
        f_hourly = self.xr2numpy(data_sel_hourly, scale=True)
        f_static = self.xr2numpy(data_sel_stat, scale=True)

        if self.return_baseline:
            data_sel_hourly_msc = data_sel[self.features_hourly_msc]
            data_sel_hourly_baseline = xr.Dataset()
            for var in data_sel_hourly_msc.data_vars:
                var_nosuffix = var.removesuffix('_msc')
                if var_nosuffix == 'tp' and self.precip_zero_baseline:
                    data_sel_hourly_baseline[var_nosuffix] = xr.full_like(data_sel_hourly[var_nosuffix], 0.0)
                else:
                    data_sel_hourly_baseline[var_nosuffix] = msc_align(data_sel_hourly_msc[var], data_sel_hourly)

            data_sel_stat_baseline = xr.Dataset()
            for var in data_sel_stat.data_vars:
                data_sel_stat_baseline[var] = xr.full_like(data_sel_stat[var], self.data[var].attrs['mean'])

            f_hourly_bl = self.xr2numpy(data_sel_hourly_baseline, scale=True)
            f_static_bl = self.xr2numpy(data_sel_stat_baseline, scale=True)
        else:
            f_hourly_bl = None
            f_static_bl = None

        if len(self.target_hourly) > 0:
            target_hourly = self.target_hourly[0]
            t_hourly_ts = self.xr2numpy(data_sel[[target_hourly]], scale=True)
            t_hourly_msc = self.xr2numpy(data_sel[[target_hourly + '_msc']], scale=True)
            t_hourly_ano = self.xr2numpy(data_sel[[target_hourly + '_ano']], scale=True)
        else:
            t_hourly_ts = None
            t_hourly_msc = None
            t_hourly_ano = None

        if len(self.target_daily) > 0:
            target_daily = self.target_daily[0]
            t_daily_ts = self.xr2numpy(data_sel[[target_daily]], scale=True)
            t_daily_msc = self.xr2numpy(data_sel[[target_daily + '_msc']], scale=True)
            t_daily_ano = self.xr2numpy(data_sel[[target_daily + '_ano']], scale=True)
        else:
            t_daily_ts = None
            t_daily_msc = None
            t_daily_ano = None

        coords = Coords(
            lat=data_sel.lat.item(),
            lon=data_sel.lon.item(),
            window_start=time.start_window_date,
            window_end=time.end_window_date,
            num_days=time.num_window_days,
            chunk=self.chunk_index,
            dayofyear=dayofyear.values
        )

        return BatchPattern(
            f_hourly=f_hourly,
            f_static=f_static,
            f_hourly_bl=f_hourly_bl,
            f_static_bl=f_static_bl,
            t_hourly_ts=t_hourly_ts,
            t_hourly_msc=t_hourly_msc,
            t_hourly_ano=t_hourly_ano,
            t_daily_ts=t_daily_ts,
            t_daily_msc=t_daily_msc,
            t_daily_ano=t_daily_ano,
            coords=coords
        )

    def get_n_next(self, n: int):

        result = []
        for _ in range(n):
            result.append(self.next())

        return result

    def xr2numpy(self, x: xr.Dataset, scale: bool, is_baseline: bool = False) -> np.ndarray | None:

        if len(x) == 0:
            return None

        if scale:
            for var in x.data_vars:
                if var not in self.data_scaling:
                    raise KeyError(
                        f'scling for variable `{var}` requested, but not present in `data_scaling`.'
                    )

                x[var] = self.normalize_var(x=x[var], stats=self.data_scaling[var], invert=False)

        return x.to_array('var').transpose(
            'time', 'dayofyear', 'hour', 'var', missing_dims='ignore').values.astype(self.dtype)

    @staticmethod
    def normalize_var(
            x: xr.DataArray | ArrayLike | Tensor,
            stats: dict[str, float],
            invert: bool = False,
            is_uncertainty: bool = False) -> xr.DataArray | ArrayLike | Tensor:
        """(De-)Normalize xarray DataArray."""

        data_mean = stats['mean']
        data_std = stats['std']

        if invert:
            if is_uncertainty:
                return x * data_std
            else:
                return x * data_std + data_mean
        else:
            if is_uncertainty:
                return x / data_std
            else:
                return (x - data_mean) / data_std

    def _get_coords(self) -> Iterable[tuple[int, DateLookup]]:

        sample_coords = np.arange(len(self.data.sample))
        time_coords = self._get_dates()

        coords = [p for p in product(sample_coords, time_coords)]
        if not self.disable_shuffling:
            np.random.shuffle(coords)
        return coords

    @staticmethod
    def get_years(data: xr.Dataset | xr.DataArray, window_size: int, context_size: int) -> tuple[np.ndarray, int, int]:

        years = np.unique(data.time.dt.year)
        num_years = len(years)

        if context_size >= num_years or context_size < 0:
            raise ValueError(
                f'{context_size=} should be in range [0, {num_years-1=}]'
            )

        if window_size < -1 or window_size == 0 or window_size > num_years:
            raise ValueError(
                f'{window_size=} should be in range [1, {num_years=}], or -1.'
            )

        if window_size == -1:
            window_size = num_years - context_size

        num_year_samples = (num_years - context_size) / window_size

        if num_year_samples % 1 != 0:
            raise ValueError(
                f'with {num_years=} of data and {context_size=}, the remaining years ({num_years-context_size=}) '
                f'must be equally dividable by the {window_size=}, which is not the case.'
            )

        num_year_samples = int(num_year_samples)

        return years, window_size, num_year_samples

    def _get_dates(self) -> list[DateLookup]:

        years, window_size, _ = self.get_years(
            data=self.data, window_size=self.window_size, context_size=self.context_size)

        date_lookup = [
            self.get_date_lookup(
                start_context_year=years[0] - self.context_size,
                start_window_year=years[0], 
                end_window_year=years[-1]
            ) for years in np.asarray(years[self.context_size:]).reshape(-1, window_size)]

        return date_lookup

    @staticmethod
    def get_date_lookup(start_context_year: int, start_window_year: int, end_window_year: int) -> DateLookup:

        num_leap_years = len([y for y in range(start_context_year, end_window_year + 1) if y % 4 == 0])
        start_context_date = f'{start_context_year}-01-{num_leap_years + 1:02d}'
        start_window_date = f'{start_window_year}-01-01'
        end_window_date = f'{end_window_year}-12-31'
        # +1 because indexing with dates in xarray includes the last date.
        num_window_days = (pd.to_datetime(end_window_date) - pd.to_datetime(start_window_date)).days + 1
        time_slice = slice(start_context_date, end_window_date)

        return DateLookup(
            start_context_year=start_context_year,
            start_window_year=start_window_year,
            end_window_year=end_window_year,
            start_context_date=start_context_date,
            start_window_date=start_window_date,
            end_window_date=end_window_date,
            num_window_days=num_window_days,
            time_slice=time_slice
        )

    @property
    def num_left(self) -> int:

        return len(self.coords) - self._current_sample


class QueueFiller(object):
    """Defines a collection of buffered chunks and how to iterate them."""

    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            data_scaling: dict[str, dict[str, float]],
            chunk_coords: xr.DataArray,
            batch_queue: mp.Queue,
            index_queue: mp.Queue,
            batch_size: int,
            chunk_buffer_size: int,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            target_hourly: list[str] = [],
            target_daily: list[str] = [],
            window_size: int = 2,
            context_size: int = 2,
            drop_last: bool = False,
            chunk_size: int = 20,
            disable_shuffling: bool = False,
            return_baseline: bool = False,
            precip_zero_baseline: bool = False,
            dummy_data: bool = False) -> None:
        """Initialize QueueFiller.

        Args:
            ds: the dataset containing spatio-temporal observations. Must at least have
                the dimensions `lat` and `lon`.
            mask: a spatio-temporal mask indicating valid pixels (`True`). Must at least have
                the dimensions `lat` and `lon` equal to `ds`.
            data_scaling: variable scaling, dicts with pattern {<variable>: {'mean': <value>, 'std': <value>}}.
            chunk_coords: coordinates corresponding to chunks with any valid samples in them,
                an array with shape n x 2, each element being a lat, lon coordinate.
            a spatio-temporal mask indicating valid chunks (`True`). Must at least have
                the dimensions `lat` and `lon`.
            batch_queue: a multiprocessing.Manager().Queue for queuing of batches.
            index_queue: a multiprocessing.Manager().Queue holding the chunk indices.
            batch_size: the batch size, an integer larger than 0.
            chunk_buffer_size: the size of the queue, i.e., max number of batches to preload.
            features_hourly: a list of hourly dynamic features, must be data variables of `ds`.
            features_static: a list of static features, must be data variables of `ds`.
            target_hourly: hourly dynamic target, must be data variables of `ds`.
            target_daily: daily dynamic target, must be data variables of `ds`.
            window_size: the window size in years (an integer) used for gradient computation during training.
                Default is 2.
            context_size: additional temporal context size in years (an integer).
            drop_last: Whether to drop the last batch with size < `batch_size`. Default is `False`.
            chunk_size: the latitude/longitude chunk sizes. Must divide the
                `ds.lat`/`ds.lon` chunk size without remainder.
            disable_shuffling: if `True` shuffling will be turned off. Default is `False`.
            return_baseline: if `True`, the baseline values for each feature are returned. This is the mean for static
                features, the seasonality for the hourly features (except tp, where the baseline is all 0.0). Default
                is `False.`
            precip_zero_baseline: if `True`, the precipitation baseline is set to 0.0, else the seasonality is taken.
                Default is `False`.
            dummy_data: if set to `True`, dummy data is returned and reading from disk is omitted; use for debugging.
                Default is `False`.

        """

        super().__init__()

        self.data = data
        self.mask = mask.load()
        self.data_scaling = data_scaling
        self.chunk_coords = chunk_coords
        self.batch_queue = batch_queue
        self.index_queue = index_queue

        self.features_hourly = features_hourly
        self.features_static = features_static
        self.target_hourly = target_hourly
        self.target_daily = target_daily
        self.window_size = window_size
        self.context_size = context_size

        self.batch_size = batch_size
        self.chunk_buffer_size = chunk_buffer_size
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.disable_shuffling = disable_shuffling
        self.return_baseline = return_baseline
        self.precip_zero_baseline = precip_zero_baseline
        self.dummy_data = dummy_data

        self.chunk_bounds_lat = self.coords2bounds(self.mask, dim='lat', chunk_size=self.chunk_size)
        self.chunk_bounds_lon = self.coords2bounds(self.mask, dim='lon', chunk_size=self.chunk_size)

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
                                    target_daily=self.target_daily,
                                    target_hourly=self.target_hourly,
                                    window_size=self.window_size,
                                    context_size=self.context_size,
                                    data_scaling=self.data_scaling,
                                    disable_shuffling=self.disable_shuffling,
                                    dummy_data=self.dummy_data,
                                    chunk_index=index,
                                    return_baseline=self.return_baseline,
                                    precip_zero_baseline=self.precip_zero_baseline
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
                self.batch_queue.put(QueueException(exception=e, traceback=traceback.format_exc()))

    def clean_empty_chunks(self) -> None:

        empty = []
        for i, chunk in enumerate(self.chunks):
            if chunk.num_left == 0:
                empty.append(i)

        for i in reversed(empty):
            self.chunks.pop(i)

    @staticmethod
    def coords2bounds(mask: xr.DataArray, dim: str, chunk_size: int):

        return np.lib.stride_tricks.sliding_window_view(range(0, len(mask[dim]) + 1, chunk_size), 2)

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
    def num_target_hourly(self) -> int:
        return len(self.target_hourly)

    @property
    def num_target_daily(self) -> int:
        return len(self.target_daily)


class DataQueue(IterableDataset):
    def __init__(
            self,
            data: xr.Dataset,
            mask: xr.DataArray,
            fold_id: int,
            batch_size: int,
            queue_size: int,
            chunk_buffer_size: int,
            num_queue_workers: int = 12,
            features_hourly: list[str] = [],
            features_static: list[str] = [],
            target_hourly: list[str] = [],
            target_daily: list[str] = [],
            window_size: int = 2,
            context_size: int = 2,
            drop_last: bool = False,
            chunk_size: int = 20,
            disable_shuffling: bool = False,
            num_split: int = 0,
            split_idx: int = 0,
            return_baseline: bool = False,
            precip_zero_baseline: bool = False,
            dummy_data: bool = False):
        super().__init__()

        self.data = data
        self.mask = mask.load()
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.chunk_buffer_size = chunk_buffer_size
        self.num_queue_workers = num_queue_workers
        self.features_hourly = features_hourly
        self.features_static = features_static
        self.target_hourly = target_hourly
        self.target_daily = target_daily
        self.window_size = window_size
        self.context_size = context_size
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.disable_shuffling = disable_shuffling
        self.num_split = num_split
        self.split_idx = split_idx
        self.return_baseline = return_baseline
        self.precip_zero_baseline = precip_zero_baseline
        self.dummy_data = dummy_data

        _, _, self.num_year_samples = DataChunk.get_years(
            data=self.data, window_size=window_size, context_size=context_size)
        self.chunk_mask = (self.mask.coarsen(lat=chunk_size, lon=chunk_size).sum() > 0).compute()
        self.chunk_coords = np.argwhere(self.chunk_mask.values)
        if self.num_split > 1:
            chunk_ids = np.arange(self.num_chunks)
            chunk_ids = np.array_split(chunk_ids, self.num_split)
            keep_ids = chunk_ids.pop(self.split_idx)
            remove_ids = np.concatenate(chunk_ids)

            remove_chunk_coords = self.chunk_coords[remove_ids]
            self.chunk_coords = self.chunk_coords[keep_ids]

            lat_chunk_coords = QueueFiller.coords2bounds(mask=self.mask, dim='lat', chunk_size=self.chunk_size)
            lon_chunk_coords = QueueFiller.coords2bounds(mask=self.mask, dim='lon', chunk_size=self.chunk_size)

            for lat_i, lon_i in remove_chunk_coords:
                self.mask[{'lat': slice(*lat_chunk_coords[lat_i]), 'lon': slice(*lon_chunk_coords[lon_i])}] = 0
                self.chunk_mask[{'lat': lat_i, 'lon': lon_i}] = 0

        self.data_scaling = self._get_scaling(
            self.data, self.features_hourly, self.features_static, self.target_hourly, self.target_daily)

        self.num_samples = self.mask.sum().compute().item() * self.num_year_samples
        self._check_ds(self.data, self.mask)

    def get_queue_filler(
            self,
            batch_queue: mp.Queue,
            index_queue: mp.Queue) -> QueueFiller:

        queue_filler = QueueFiller(
            data=self.data,
            mask=self.mask,
            data_scaling=self.data_scaling,
            chunk_coords=self.chunk_coords,
            batch_queue=batch_queue,
            index_queue=index_queue,
            batch_size=self.batch_size,
            chunk_buffer_size=self.chunk_buffer_size,
            features_hourly=self.features_hourly,
            features_static=self.features_static,
            target_daily=self.target_daily,
            target_hourly=self.target_hourly,
            window_size=self.window_size,
            context_size=self.context_size,
            drop_last=False,
            disable_shuffling=self.disable_shuffling,
            return_baseline=self.return_baseline,
            precip_zero_baseline=self.precip_zero_baseline,
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

        return processes

    def __iter__(self):

        manager = mp.Manager()
        batch_queue = manager.Queue(maxsize=self.queue_size)
        index_queue = manager.Queue()

        processes = self.init_queue_fillers(
            batch_queue=batch_queue,
            index_queue=index_queue,
            num_processes=self.num_queue_workers
        )

        with GracefulExit(processes=processes, manager=manager):

            if self.disable_shuffling:
                chunk_ids = np.arange(self.num_chunks)
            else:
                chunk_ids = np.random.permutation(self.num_chunks)

            for i in chunk_ids:
                index_queue.put(i)

            for process in processes:
                process.start()

            subsize_batches = []

            has_more_batches = True

            while has_more_batches:
                try:
                    el: BatchPattern | QueueException = batch_queue.get_nowait()

                    if isinstance(el, QueueException):
                        print('An exception was raised in the QueueFiller:')
                        print(el[1])
                        raise el[0]
                    elif not isinstance(el, BatchPattern):
                        raise TypeError(
                            'retreived element from queue which is neither a `BatchPattern` nor a '
                            f' `QueueException`, but of type `{type(el).___name__}`'
                        )

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
                        *[el if el is None else el[i] for el in incomplete_batch[:10]],
                        coords=Coords(
                            lat=incomplete_batch.coords.lat[i],
                            lon=incomplete_batch.coords.lon[i],
                            window_start=incomplete_batch.coords.window_start[i],
                            window_end=incomplete_batch.coords.window_end[i],
                            num_days=incomplete_batch.coords.num_days[i],
                            chunk=incomplete_batch.coords.chunk[i],
                            dayofyear=incomplete_batch.coords.dayofyear[i]))
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
            f_hourly=torch.randn(self.batch_size, 730, 24, len(self.features_hourly)) if self.features_hourly else None,
            f_static=torch.randn(self.batch_size, len(self.features_static)) if self.features_static else None,
            f_hourly_bl=torch.randn(self.batch_size, 730, 24, len(self.features_hourly)) if self.features_hourly else None,
            f_static_bl=torch.randn(self.batch_size, len(self.features_static)) if self.features_static else None,
            t_daily=torch.randn(self.batch_size, 730, len(self.target_daily)) if self.target_daily else None,
            t_hourly=torch.randn(self.batch_size, 730, 24, len(self.target_hourly)) if self.target_hourly else None,
            coords=Coords(
                lat=torch.arange(self.batch_size), lon=torch.arange(self.batch_size),
                window_start=['2006-01-01'] * self.batch_size,
                window_end=['2006-12-31'] * self.batch_size,
                num_days=[365] * self.batch_size,
                chunk=torch.arange(self.batch_size),
                dayofyear=np.arange(len(730)) % 366
            )
        )

    @property
    def target(self) -> list[str]:
        return self.target_daily + self.target_hourly

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_coords)

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

    def _get_scaling(
            self,
            data: xr.Dataset,
            features_hourly: list[str],
            features_static: list[str],
            target_hourly: list[str],
            target_daily: list[str]) -> dict[str, dict[str, float]]:

        stats = {}

        if len(target_hourly) > 0:
            target_hourly = [target_hourly[0], target_hourly[0] + '_msc', target_hourly[0] + '_ano']
        if len(target_daily) > 0:
            target_daily = [target_daily[0], target_daily[0] + '_msc', target_daily[0] + '_ano']

        for var in features_hourly + features_static + target_hourly + target_daily:

            data_mean = data[var.removesuffix('_msc')].attrs['mean']
            data_std = data[var.removesuffix('_msc')].attrs['std']
            stats.update({var: {'mean': data_mean, 'std': data_std}})

        return stats

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
    > BatchPattern = namedtuple(
        f_hourly: Tensor
        f_static: Tensor
        f_hourly_bl: Tensor | None
        f_static_bl: Tensor | None
        t_hourly_ts: Tensor
        t_hourly_msc: Tensor
        t_hourly_ano: Tensor
        t_daily_ts: Tensor
        t_daily_msc: Tensor
        t_daily_ano: Tensor
        coords: Coords
    )
    The `coords` are formatted as:
    > Coords = namedtuple(
        lat: int
        lon: int
        chunk: int
        window_start: str
        window_end: str
        num_days: int
        dayofyear: list[int]
    )

    `f_hourly` are hourly features. They have shape [B, H, D, FH]
    `f_static` are static features. They have shape [B, FS]
    `f_hourly_bl` are hourly feature baselines. Same shape as `f_hourly`
    `f_static_bl` are static feature baselines. Same shape as `f_static`
    `t_hourly_ts` is hourly target. They have shape [B, H, D, TH]
    `t_hourly_msc` same as `t_hourly_ts` but seasonality
    `t_hourly_ano` same as `t_hourly_ts` but anomalies
    `t_daily` is daily target. They have shape [B, D, TD]
    `t_daily_msc` same as `t_daily` but seasonality
    `t_daily_ano` same as `t_daily` but anomalies
    `lat`, `lon` are latitudes and longitudes of each samples, they both have shape [B]
    `chunk` is the chunk id of each samples with shape [B]
    `window_start` is the time stamp of the start date
    `window_end` is the time stamp of the end date
    `num_days` is the number of days the model predicts (not including additional temp. context)
    `dayofyear` day of year for the date (D) dimension

    * batch_size: B
    * num_hours: H (=24)
    * num_days: D
    * num_features_hourly: FH
    * num_features_static: FS
    * num_target_hourly: TH
    * num_target_daily: TD

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
            target_daily: str | None = None,
            target_hourly: str | None = None,
            batch_size: int = 50,
            chunk_buffer_size: int = 4,
            queue_size: int = 10,
            num_queue_workers: int = 12,
            chunk_size: int = 20,
            window_size: int = 2,
            context_size: int = 2,
            full_seq_prediction: bool = True,
            n_mask_erode: int = 0,
            disable_shuffling: bool = False,
            return_baseline: bool = False,
            precip_zero_baseline: bool = False,
            num_split: int = 0,
            split_idx: int = -1,
            full_predict_loader: bool = False,
            cube_path: str = '/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr',
            dummy_data: bool = False,
            **dataloader_kwargs):
        """GeoDataQueue initialization.

        Notes on sequence handling:
            The model receives `context_size` + `window_size` years from a time series. The `context_size` years
            are dropped from gradient computation, whihc is only done for the `window_size` years. The sequence
            length is adjusted for leap years, the days are cut in the context. Note that this leads to cutting
            values from the moving window if `context_size` is 0. For validation/testing/prediction, the window
            size corresponds to the entire sequence except the temporal context if `full_seq_prediction` is True.
            Else, the same windowing as in training is used.

        Args:
            fold_id: the fold ID, an integer in the range (0, 3).
            target_daily: the daily target. Can eighter
                - not be passed (then `target_hourly` must be passed),
                - be length `1` (then `target_hourly` must either not be passed or also be length `1`),
                - or be length `2` (then `target_hourly` must either not be passed or also be length `2`).
                With two targets, the esecond one is assumed to be the mean seasonal cycle.
            target_hourly: the hourly target(s). Not implemented yet, takes no effect. Also
                see `target_daily`.
            batch_size: the batch size, an integer > 0. Default is 50.
            chunk_buffer_size: the number of spatial chunks to be buffered, an integer > 0.
                Note that smaller values lead to less randomness in the samples (they tend to come from
                the same spatial chunk for continuous batches), and large values lead to long initialization
                time. Values between 3 and 6 are suggested. Default is 4.
            queue_size: the number of batches to queue up. Large values can lead to memory issues. Default is 10.
            num_queue_workers: number of workers that fill batch queue in parallel.
            chunk_size: the latitude/longitude chunk sizes. Must divide the
                `ds.lat`/`ds.lon` chunk size without remainder.
            window_size: the window size in years (an integer) used for gradient computation during training.
                Default is 2.  See 'Notes on sequence handling' for more information.
            context_size: additional temporal context size in years (an integer). Default is 2. See 'Notes on
                sequence handling' for more information.
            full_seq_prediction: whether to validate/test/predict the full time-series at once (default, True) or
                to use snippets of length `window_size`. If True, the time series length changes in inference, but
                the forward run may be much faster.
            n_mask_erode: if > 1, training grid cells with a distance < `n_mask_erode` will be removed from
                the mask border to reduce dependency between CV folds. Values of 0 and 1 turn off erosion,
                default is 0.
            return_baseline: if `True`, the baseline values for each feature are returned. This is the mean for static
                features, the seasonality for the hourly features (except tp, where the baseline is all 0.0). Default
                is `False.`
            precip_zero_baseline: if `True`, the precipitation baseline is set to 0.0, else the seasonality is taken.
                Default is `False`.
            num_split: optional integer n > 1 to split data chunks into n sets. This allows to create chunk-independent
                dataloaders. Default is 0, meaning all the chunks are processed in a single dataloader.
            split_idx: when `num_split` > 1, the `split_idx` determines wich split is used. Must be an integer in the
                range 0 to `num_split` - 1.
            full_predict_loader: if `True`, all chunks are predicted (not only the predict/test set). Default is `False`.
            cube_path: the path to the data cube (zarr format).
                Default is '/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr'.
            dummy_data: if set to `True`, dummy data is returned and reading from disk is omitted; use for debugging.
                Default is `False`.
            dataloader_kwargs: keyword arguments passed to `torch.Dataset(...)`
        """

        super().__init__()

        self.fold_id = fold_id
        self.batch_size = batch_size
        self.chunk_buffer_size = chunk_buffer_size
        self.queue_size = queue_size
        self.num_queue_workers = num_queue_workers
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.context_size = context_size
        self.full_seq_prediction = full_seq_prediction
        self.n_mask_erode = n_mask_erode
        self.disable_shuffling = disable_shuffling
        self.return_baseline = return_baseline
        self.precip_zero_baseline = precip_zero_baseline
        if (num_split > 1) and ((split_idx > num_split - 1) or (split_idx < 0)):
            raise ValueError(
                f'misconfiguration: `split_idx` must in range 0 to `num_split` - 1 (0-{num_split - 1}), is {split_idx}.'
            )
        self.num_split = num_split
        self.split_idx = split_idx
        self.full_predict_loader = full_predict_loader
        self.dummy_data = dummy_data
        self.dataloader_kwargs = dataloader_kwargs

        self.cube_path = cube_path
        self.ds = xr.open_zarr(self.cube_path)

        self.train_folds, self.valid_folds, self.test_folds = \
            self.get_fold_split(fold_id=self.fold_id)

        self.fold_mask = self.ds.fold_mask.load()

        self.features_hourly = [
            'rh_cf', 'ssrd', 't2m', 'tp'
        ]
        self.features_static = [
            'canopyheight', 'rootdepth', 'percent_tree_cover', 'sandfrac', 'topidx', 'wtd'
        ]

        if target_daily is None and target_hourly is None:
            raise ValueError(
                'pass either `target_daily` or `target_hourly`.'
            )

        self.target_daily = [] if target_daily is None else [target_daily]
        self.target_hourly = [] if target_hourly is None else [target_hourly]

    @staticmethod
    def get_fold_split(fold_id: int, num_folds: int = 10) -> tuple[list[int], list[int], list[int]]:

        if (fold_id < 0) or (fold_id >= num_folds):
            raise ValueError(
                f'`fold_id` out of range; must be an integer in the range [0, {num_folds-1}]'
            )

        folds = set(range(1, num_folds + 1))

        val_fold = {(fold_id) % num_folds + 1}
        test_fold = {(fold_id + 1) % num_folds + 1}
        folds = folds - val_fold - test_fold
        folds = list(folds)

        return folds, list(val_fold), list(test_fold)

    def get_dataloader(self, cvset: str):

        # The default behavior:
        chunk_buffer_size = self.chunk_buffer_size
        num_queue_workers = self.num_queue_workers
        disable_shuffling = True
        window_size = -1 if self.full_seq_prediction else self.window_size

        # The cvset-specific behavior:
        if cvset == 'train':
            mask = self.fold_mask.isin(self.train_folds)

            if self.n_mask_erode > 1:
                ref_mask = self.fold_mask.isin(self.valid_folds + self.test_folds)
                mask = self.erode_mask(ref_mask=ref_mask, erode_mask=mask, n=self.n_mask_erode)

            window_size = self.window_size

            if self.trainer is None or self.trainer.overfit_batches == 0:
                disable_shuffling = False  # Only needed for training.
            else:
                num_queue_workers = 1  # For consistent batch.

        elif cvset == 'valid':
            mask = self.fold_mask.isin(self.valid_folds)

            chunk_buffer_size = 1  # No randomness needed.

        elif cvset == 'test':
            mask = self.fold_mask.isin(self.test_folds)
            chunk_buffer_size = 1  # No randomness needed.

        elif cvset == 'predict':
            if self.full_predict_loader:
                predict_folds = self.train_folds + self.valid_folds + self.test_folds
            else:
                predict_folds = self.test_folds
            mask = self.fold_mask.isin(predict_folds)
            chunk_buffer_size = 1  # No randomness needed.

        else:
            raise ValueError(
                f'`cvset`  must be one of \'train\', \'valid\', \'test\', or \'predict\',  is \'{cvset}\'.'
            )

        dataqueue = DataQueue(
            data=self.ds,
            mask=mask,
            fold_id=self.fold_id,
            batch_size=self.batch_size,
            num_queue_workers=num_queue_workers,
            queue_size=self.queue_size,
            chunk_buffer_size=chunk_buffer_size,
            features_hourly=self.features_hourly,
            features_static=self.features_static,
            target_daily=self.target_daily,
            target_hourly=self.target_hourly,
            window_size=window_size,
            context_size=self.context_size,
            drop_last=False,
            disable_shuffling=disable_shuffling,
            num_split=self.num_split,
            split_idx=self.split_idx,
            return_baseline=self.return_baseline,
            precip_zero_baseline=self.precip_zero_baseline,
            dummy_data=self.dummy_data
        )

        dataloader = DataLoader(
            dataqueue,
            # Do not change any of these:
            num_workers=0,  # Cannot be > 0 as we spawn subprocesses in dataqueue in 'fork' mode.
            batch_size=None,  # We manually combine batches.
            shuffle=False,  # Has no impact as we use a custom IterableDataset.
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

    @staticmethod
    def erode_mask(ref_mask, erode_mask, n):

        if n % 2 != 1:
            raise ValueError(
                f'kernel size must be an odd number, is {n}.'
            )

        r = n // 2

        y, x = np.ogrid[-r:n - r, -r:n - r]
        mask = x * x + y * y <= r * r

        kernel = np.zeros((n, n), dtype=bool)
        kernel[mask] = True

        mask_ = ref_mask.copy()
        mask_.values[:] = binary_dilation(ref_mask.values, structure=kernel)
        eroded_mask = ~mask_ & erode_mask
        return eroded_mask

    @property
    def num_features_hourly(self) -> int:
        return len(self.features_hourly)

    @property
    def num_features_static(self) -> int:
        return len(self.features_static)

    @property
    def targets(self) -> list[str]:
        return self.target_daily + self.target_hourly

import multiprocessing.pool as mpp
from multiprocessing import Pool
import tqdm
import numpy as np
import xarray as xr
import dask

from typing import Callable

dask.config.set(scheduler='synchronous')


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def chunk_parcall(
        path: str,
        var: str,
        num_processes: int,
        fun: Callable,
        use_mask: bool,
        desc: str = 'Processing |',
        args: tuple = ()):
    """Apply function parallelized over data chunks.

    Args:
        path: path to .zarr data cube.
        var: variable from cube.
        num_processes: number of processes.
        fun: A function with signature:
            if use_mask:
                path, var, mask, lat_slice, lon_slice
            else:
                path, var, lat_slice, lon_slice
        use_mask: if True, only non-masked chunks are processed.
        desc: the tqdm description to desplay before the variable name. Default is
            'Processing |', which displays as 'Processing | my_var'.
        args: tuple passed to `fun(..., *args)` as last positional agruments.

    Returns:
        A list of return values from `fun`.

    """
    cube = xr.open_zarr(path)

    if use_mask:
        mask = cube.mask.load()

    da = cube[var]

    lat_chunk_bounds = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros(1, dtype=int), np.cumsum(da.chunksizes['lat']))), 2)
    lon_chunk_bounds = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros(1, dtype=int), np.cumsum(da.chunksizes['lon']))), 2)

    iterable = []

    for lat_chunk_bound in lat_chunk_bounds:
        for lon_chunk_bound in lon_chunk_bounds:
            lat_slice = slice(*lat_chunk_bound)
            lon_slice = slice(*lon_chunk_bound)

            if use_mask:
                if mask.isel(lat=lat_slice, lon=lon_slice).sum() > 0:
                    iterable.append((path, var, mask, lat_slice, lon_slice, *args))
            else:
                iterable.append((path, var, lat_slice, lon_slice, *args))

    results = []
    with Pool(num_processes) as pool:
        for r in tqdm.tqdm(
                pool.istarmap(fun, iterable),
                total=len(iterable),
                desc=f'{desc} {var:<18}',
                miniters=200):
            results.append(r)

    return results

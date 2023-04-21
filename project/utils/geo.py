
import xarray as xr
import numpy as np
import pandas as pd
from typing import overload


def get_slice_from_anchor(
        ds: xr.DataArray,
        lon: int,
        lat: int,
        num_chunks_lon: int = 12,
        num_chunks_lat: int = 6,
        chunk_size_lon: int = 10,
        chunk_size_lat: int = 10) -> dict[str, slice]:
    # lon: bottom left, lat: bottom left, round to closest chunk border

    chunk_edge_lon = ds.lon[::chunk_size_lon]
    chunk_edge_lat = ds.lat[::chunk_size_lat]

    lon_min = lon
    lat_min = lat

    left_idx = chunk_edge_lon.get_index('lon').get_indexer([lon_min], method='nearest').item() * chunk_size_lon
    bot_idx = chunk_edge_lat.get_index('lat').get_indexer([lat_min], method='nearest').item() * chunk_size_lat

    right_idx = left_idx + chunk_size_lon * num_chunks_lon
    top_idx = bot_idx - chunk_size_lat * num_chunks_lat

    lon_min = ds.lon.values[left_idx]
    lon_max = ds.lon.values[right_idx - 1]
    lat_min = ds.lat.values[bot_idx - 1]
    lat_max = ds.lat.values[top_idx]

    return {
        'lon': slice(lon_min - 0.01, lon_max + 0.01),
        'lat': slice(lat_max + 0.01, lat_min - 0.01)
    }


def sel_to_poly_corners(sel: dict[slice]) -> list[list[int]]:
    corners = [
        [sel['lon'].stop, sel['lat'].stop],
        [sel['lon'].stop, sel['lat'].start],
        [sel['lon'].start, sel['lat'].start],
        [sel['lon'].start, sel['lat'].stop],
    ]
    return corners


@overload
def stacktime(ds: xr.Dataset, dim: str) -> xr.Dataset:
    ...


@overload
def stacktime(ds: xr.DataArray, dim: str) -> xr.DataArray:
    ...


def stacktime(ds: xr.Dataset | xr.DataArray, dim: str = 'time') -> xr.Dataset | xr.DataArray:
    """Stack time (day) x hour into single time dimension.

    Args:
        ds: data to reformat with temporal dimensions 'time' and 'hour'.
        dim: time dimension name, default is 'time'.
    Returns:
        ds with same type as input (xr.Dataset or xr.DataArray) with new dimension 'time'
            and dropped dimension 'hour'.

    """
    for required_dim in [dim, 'hour']:
        if required_dim not in ds.dims:
            raise KeyError(f'required dimension \'{required_dim}\' not found in `ds` with dimensions {ds.dims}.')

    dsstacked = ds.stack(t=(dim, 'hour'))
    dsstacked['timvals'] = dsstacked[dim] + dsstacked.hour.astype('timedelta64[h]')

    return dsstacked.set_index(t='timvals').rename({dim: 'old_time'}).rename(t=dim).drop_vars('old_time')


@overload
def stackdims(ds: xr.Dataset, dims: list[str], new_dim: str, start_at_zero: bool) -> xr.Dataset:
    ...


@overload
def stackdims(ds: xr.DataArray, dims: list[str], new_dim: str, start_at_zero: bool) -> xr.DataArray:
    ...


def stackdims(
        ds: xr.Dataset | xr.DataArray,
        dims: list[str],
        new_dim: str,
        start_at_zero: bool = True) -> xr.Dataset | xr.DataArray:
    """Stack dimensions into single dimension.

    Args:
        ds: data to reformat with at least dimensions `dims`.
        dims: dimension names to stack.
        new_dim: new dimension name.
        start_at_zero: whether to assign new coordinates from 0 to n-1 (True, default), or
            from -n + 1 to 0.
    Returns:
        ds with same type as input (xr.Dataset or xr.DataArray) with new dimension `new_dim`
            and dropped dimensions `dims`.

    """
    for required_dim in dims:
        if required_dim not in ds.dims:
            raise KeyError(f'required dimension \'{required_dim}\' not found in `ds` with dimensions {ds.dims}.')

    dsstacked = ds.stack({'_temp_dim_': dims}).reset_index('_temp_dim_').drop(dims).rename({'_temp_dim_': new_dim})
    n = len(dsstacked[new_dim])
    if start_at_zero:
        return dsstacked.assign_coords({new_dim: np.arange(0, n)})
    else:
        return dsstacked.assign_coords({new_dim: np.arange(-n + 1, 1)})


@overload
def msc2date(ds: xr.Dataset) -> xr.Dataset:
    ...


@overload
def msc2date(ds: xr.DataArray) -> xr.DataArray:
    ...


def msc2date(ds):
    """Change seasonal data with dayofyear x hour into single time coordinate (with year 2000 as base).

    Args:
        ds: data to reformat with temporal dimension 'dayofyear'.

    Returns:
        ds with same type as input (xr.Dataset or xr.DataArray) with new dimension 'time' and
            dropped dimension 'dayofyear' (and 'hour' if present in `ds`).

    # TODO: Handle data without hour dimension.

    """
    if 'dayofyear' not in ds.dims:
        raise KeyError(f'required dimension \'dayofyear\' not found in `ds` with dimensions {ds.dims}.')

    if 'hour' in ds.dims:
        dsstacked = ds.stack(time=('dayofyear', 'hour')).drop('hour')
        return dsstacked.assign_coords(
            time=pd.date_range('2000-01-01', '2001-01-01', freq='H', inclusive='left'))
    else:
        return ds.assign_coords(
            dayofyear=pd.date_range(
                '2000-01-01', '2001-01-01', freq='D', inclusive='left')).rename({'dayofyear': 'time'})


@overload
def msc_align(msc: xr.Dataset, ref: xr.Dataset | xr.DataArray) -> xr.Dataset:
    ...


@overload
def msc_align(msc: xr.DataArray, ref: xr.Dataset | xr.DataArray) -> xr.DataArray:
    ...


def msc_align(msc: xr.Dataset | xr.DataArray, ref: xr.Dataset | xr.DataArray):
    """Align seasonality with dayofyear with reference data time dimension.

    Args:
        msc: seasonality data to reformat with temporal dimension 'dayofyear', and optionally 'hour'.
        ref: reference data with temporal dimension `time`.

    Returns:
        ds with same type as msc input (xr.Dataset or xr.DataArray)

    """
    if 'dayofyear' not in msc.dims:
        raise KeyError(f'required dimension \'dayofyear\' not found in `msc` with dimensions {msc.dims}.')

    for required_var in ['time']:
        if required_var not in ref.dims:
            raise KeyError(f'required dimension \'{required_var}\' not found in `msc` with dimensions {ref.dims}.')

    msc_aligned = msc.sel(dayofyear=ref.time.dt.dayofyear - 1)
    return msc_aligned.chunk({'time': 1000})

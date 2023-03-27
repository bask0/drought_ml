from argparse import ArgumentParser
import xarray as xr
import numpy as np
import dask

from preprocessing.chunk_parallel import chunk_parcall

dask.config.set(scheduler='synchronous')


def calculate_anomaly(x: xr.DataArray, msc_only: bool = False) -> tuple[xr.DataArray, xr.DataArray | None]:
    gb = x.groupby('time.dayofyear')
    msc = gb.mean('time').compute()

    msc_0 = msc.copy().assign_coords(dayofyear=np.arange(1 - 366, 1))
    msc_1 = msc.copy().assign_coords(dayofyear=np.arange(367, 367 + 366))
    msc_stack = xr.concat((msc_0, msc, msc_1), dim='dayofyear')
    msc_smooth = msc_stack.rolling(dayofyear=17, min_periods=5, center=True).mean().sel(dayofyear=slice(1, 366))

    if msc_only:
        anomalies = None
    else:
        anomalies = gb - msc_smooth
        anomalies = anomalies.drop('dayofyear')

    return msc_smooth, anomalies


def batch_calculate_anomaly(
        path: str,
        var: str,
        lat_subset: slice,
        lon_subset: slice,
        msc_only: bool) -> None:

    cube = xr.open_zarr(path)
    da = cube[var].isel(lat=lat_subset, lon=lon_subset).load()

    msc, ano = calculate_anomaly(da, msc_only=msc_only)

    # Seasonality
    msc_da = xr.Dataset()
    msc_da[f'{var.lower()}_msc'] = msc

    if 'hour' in msc_da.dims:
        msc_da = msc_da.drop_vars(['hour', 'dayofyear', 'lat', 'lon'])
    else:
        msc_da = msc_da.drop_vars(['dayofyear', 'lat', 'lon'])

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        msc_da.to_zarr(path, consolidated=False, region={
            'lat': lat_subset,
            'lon': lon_subset
        })

    if not msc_only:
        # Anomalies
        ano_da = xr.Dataset()
        ano_da[f'{var.lower()}_ano'] = ano

        if 'hour' in ano_da.dims:
            ano_da = ano_da.drop_vars(['hour', 'time', 'lat', 'lon'])
        else:
            ano_da = ano_da.drop_vars(['time', 'lat', 'lon'])

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ano_da.to_zarr(path, consolidated=False, region={
                'lat': lat_subset,
                'lon': lon_subset
            })


def calculate_anomalies(
        path: str,
        var: str,
        num_processes: int = 1,
        msc_only: bool = True):

    desc = 'Compute MSC |' if msc_only else 'Compute MSC & ANO |'
    chunk_parcall(
        path=path,
        var=var,
        num_processes=num_processes,
        fun=batch_calculate_anomaly,
        use_mask=False,
        desc=desc,
        args=(msc_only,)
    )


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default='./test.zarr',
        help='dataset path (end with .zarr)')
    parser.add_argument(
        '-v',
        '--variable',
        type=str,
        help='variable for whihc to compute decomposition.')
    parser.add_argument(
        '--msc_only',
        action='store_true',
        help='whether to only calculate MSC, or both MSC and ANO.')
    parser.add_argument(
        '-n',
        '--num_processes',
        type=int,
        help='number of processes')

    args = parser.parse_args()

    calculate_anomalies(
        path=args.path,
        var=args.variable,
        num_processes=args.num_processes,
        msc_only=args.msc_only)

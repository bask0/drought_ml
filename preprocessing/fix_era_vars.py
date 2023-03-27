from argparse import ArgumentParser
import xarray as xr
import zarr
import numpy as np
import dask

from preprocessing.chunk_parallel import chunk_parcall

dask.config.set(scheduler='synchronous')


def c2flux(
        path: str,
        var: str,
        lat_subset: slice,
        lon_subset: slice) -> None:

    cube = xr.open_zarr(path)
    da = cube[var].isel(lat=lat_subset, lon=lon_subset).load()
    da_flat = da.values.reshape(-1, *da.shape[-2:])
    da_flat_diff = np.diff(da_flat, axis=0, append=0)
    da_flat_diff[0::24] = da_flat[1::24]
    da_flat_diff[-1, ...] = 0.0
    da.values = da_flat_diff.reshape(-1, 24, *da.shape[-2:])

    da = da.drop_vars(['hour', 'time', 'lat', 'lon'])
    ds = xr.Dataset()
    ds[var] = da

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds.to_zarr(path, consolidated=True, region={
            'lat': lat_subset,
            'lon': lon_subset
        })


def fix_vars(path: str, vars: list[str] = ['tp', 'ssrd'], num_processes: int = 1):

    cube = xr.open_zarr(path)

    for var in vars:
        if var not in cube.data_vars:
            raise KeyError(f'variable \'{var}\' not found in cube.')

        chunk_parcall(
            path=path,
            var=var,
            num_processes=num_processes,
            fun=c2flux,
            use_mask=False,
            desc='Fixing flux variable |'
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
        '--variables',
        nargs='+',
        default=['tp', 'ssrd'],
        help='variables to convert from cumulative to flux. Default is \'tp\' and \'ssrd\'.')
    parser.add_argument(
        '-n',
        '--num_processes',
        type=int,
        help='number of processes')

    args = parser.parse_args()

    fix_vars(path=args.path, vars=args.variables, num_processes=args.num_processes)

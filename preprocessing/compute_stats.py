from argparse import ArgumentParser
import xarray as xr
import zarr
import dask

from preprocessing.chunk_parallel import chunk_parcall

dask.config.set(scheduler='synchronous')


def batch_stats(
        path: str,
        var: str,
        mask: xr.DataArray,
        lat_subset: slice,
        lon_subset: slice) -> None:

    da = xr.open_zarr(path)[var.lower()].isel(lat=lat_subset, lon=lon_subset).load()
    da = da.where(mask).load()
    counts = da.notnull().sum().compute().item()
    sums = da.sum().compute().item()
    sq_sums = (da ** 2).sum().compute().item()

    return counts, sums, sq_sums


def add_stats(path: str, variables: list[str], num_processes: int):

    cube = xr.open_zarr(path)
    z = zarr.open(path)

    for var in variables:

        results = chunk_parcall(
            path=path,
            var=var,
            num_processes=num_processes,
            fun=batch_stats,
            use_mask=True,
            desc='Computing stats |'
        )

        if len(results) == 0:
            raise RuntimeError(
                'no stats returned.'
            )

        n = 0.
        s = 0.
        s2 = 0.

        for n_, s_, s2_ in results:
            n += n_
            s += s_
            s2 += s2_

        mn = s / n
        sd = ((s2 / n) - (s / n) ** 2) ** 0.5

        z[var].attrs['mean'] = mn
        z[var].attrs['std'] = sd

    zarr.consolidate_metadata(path)


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
        help='variables for which to compute stats.')
    parser.add_argument(
        '-n',
        '--num_processes',
        type=int,
        help='number of processes')

    args = parser.parse_args()

    add_stats(path=args.path, variables=args.variables, num_processes=args.num_processes)

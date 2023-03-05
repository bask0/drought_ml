
import xarray as xr
import dask
import pandas as pd
import numpy as np
import zarr
import netCDF4 as nc
from glob import glob
import warnings
import math
from typing import Union
from tqdm import tqdm
import multiprocessing as mpl
import os

CHUNKS = {'time': 300, 'lat': 20, 'lon': 20}
CHUNKS_MSC = {'dayofyear': 366, 'lat': 20, 'lon': 20}
CHUNKS_HOURLY = {'time': 300, 'hour': 24, 'lat': 20, 'lon': 20}
CHUNKS_HOURLY_MSC = {'dayofyear': 366, 'hour': 24, 'lat': 20, 'lon': 20}
CHUNKS_ORDER = ('time', 'dayofyear', 'hour','lat', 'lon', 'layer',)


def get_data_paths(
        variables: str = '*',
        scales: str = '*',
        years: str = '*',
        months: str = '*',
        data_path: str = '/Net/Groups/BGI/scratch/bkraft/drought_data/preproc',
        error_on_empty: bool = True) -> list[str]:
    if isinstance(variables, str):
        variables = [variables]
    if isinstance(scales, str):
        scales = [scales]
    if isinstance(years, str):
        years = [years]
    if isinstance(months, str):
        months = [months]

    paths = []
    for var in variables:
        for scale in scales:
            if scale == 'static':
                paths.extend(glob(f'{data_path}/{var}.{scale}.1460.1140.nc'))
            else:
                for year in years:
                    for month in months:
                        paths.extend(glob(f'{data_path}/{var}.{scale}.1460.1140.{month}.{year}.nc'))

    if (not paths) and (error_on_empty):
        raise RuntimeError(
            f'no matches found for the query <{variables=}, {scales=}, {years=}, {months=}, {data_path=}>. '
            'To disable this error message, use `get_data_paths(..., error_on_empty=False)`.'
        )

    return paths


def get_scale_and_offset(var: str) -> dict[str, float]:
    """Calculate packing values from -30000 to 30000. The int16 range
    is not entirely used to avoid out-of-range conflices when appending data
    with potentially lower or larger values.

    Warning: as some min/max attributes are missing after preprocessing (bug?),
    some missings are tolerated. We There is a buffer for potential outliers
    in the min/max values in files not considered. Maximum 10% of the files
    are allowed to have missing attributes, else, an error is raised.
    """

    if var.lower() == 'fvc_msc':
        var = 'fvc'

    if var.lower() == 'lst_msc':
        var = 'lst'

    if var.lower() == 'fvc_ano':
        data_min = -1.
        data_max = 1.

    elif var.lower() == 'lst_ano':
        data_min = -100.
        data_max = 100.
    
    else:
        data_min, data_max = math.inf, -math.inf

        paths = get_data_paths(variables=[var])

        num_vars = len(paths)
        num_attrs_missing = 0

        for path in paths:

            ds = nc.Dataset(path, 'r', format='NETCDF4')
            try:
                min_val = ds.getncattr('min_val')
                max_val = ds.getncattr('max_val')
            except Exception as e:
                num_attrs_missing += 1
                perc_missing = num_attrs_missing / num_vars * 100
                warnings.warn(
                    f'Dataset has no attribute `min_val` or `max_val`: {path}\n'
                    f'({perc_missing:0.1f}% of maximum 10%)'
                )
                if perc_missing > 10:
                    raise AssertionError(
                        f'More than 10% of the files ({num_vars} in total) do not have attributes '
                        '`min_val` or `max_val`.'
                    )
            data_min = min(min_val, data_min)
            data_max = max(max_val, data_max)

            ds.close()

    # stretch/compress data to the available packed range
    scale_factor = (data_max - data_min) / 60000
    # translate the range to be symmetric about zero
    add_offset = data_min + 30000 * scale_factor

    return {'scale_factor': scale_factor, 'add_offset': add_offset}, data_min, data_max


def rename(x: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    return x.rename({'latitude': 'lat', 'longitude': 'lon'})


def rename_to_lower(x: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    if isinstance(x, xr.Dataset):
        for var in x.data_vars:
            x = x.rename({var: var.lower()})
    else:
        x.name = x.name.lower()

    return x


def add_stats_group(path: str, group_name: str):
    z = zarr.open(path)
    z.create_group(group_name.lower() + '_stats')


def add_stats(path: str, group_name: str, array_name: str, array: xr.DataArray):
    z = zarr.open(path)
    g = z[group_name.lower() + '_stats']

    if isinstance(array, xr.Dataset):
        g[array_name] = [
            array.notnull().sum().compute().to_array().item(),
            array.sum().compute().to_array().item(),
            (array ** 2).sum().compute().to_array().item(),
        ]
    else:
        g[array_name] = [
            array.notnull().sum().compute().item(),
            array.sum().compute().item(),
            (array ** 2).sum().compute().item(),
        ]


def hourly2dayhour(x: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    x.coords['time'] = pd.MultiIndex.from_arrays(
        arrays=[pd.to_datetime(x.time.dt.date), x.time.dt.hour.values],
        names=('date', 'hour'))

    return x.unstack('time').rename({'date': 'time'})


def create_dummy(start_year: int, end_year: int, out_path: str) -> None:

    times = pd.date_range(str(start_year), str(end_year + 1), freq='D', inclusive='left')
    dayofyear = np.arange(366)
    hours = np.arange(24)

    dummy = xr.Dataset()

    def rename(x):
        return x.rename({'latitude': 'lat', 'longitude': 'lon'})

    variables = []

    for path in get_data_paths(scales=['hourly'], years=['2004'], months=['01']):
        ds_orig = xr.open_dataset(path)
        var = list(ds_orig.data_vars)[0]

        ds = ds_orig.chunk().pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(
            time=times, hour=hours)[var].pipe(rename).transpose(*CHUNKS_ORDER, missing_dims='ignore')
        dummy[var.lower()] = ds
        variables.append(var)

        if var.lower() == 'lst':
            dummy['lst_msc'] = ds_orig.isel(time=0, drop=True).expand_dims(
                dayofyear=dayofyear, hour=hours)[var].pipe(rename).transpose(*CHUNKS_ORDER, missing_dims='ignore')
            variables.append('lst_msc')
            dummy['lst_ano'] = ds
            variables.append('lst_ano')

    for path in get_data_paths(scales=['daily'], years=['2004'], months=['01']):
        ds_orig = xr.open_dataset(path)
        var = list(ds_orig.data_vars)[0]
        ds = ds_orig.chunk().pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(
            time=times)[var].pipe(rename).transpose(*CHUNKS_ORDER, missing_dims='ignore')
        dummy[var.lower()] = ds
        variables.append(var)

        if var.lower() == 'fvc':
            dummy['fvc_msc'] = ds_orig.chunk().pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(
                dayofyear=dayofyear)[var].pipe(rename).transpose(*CHUNKS_ORDER, missing_dims='ignore')
            variables.append('fvc_msc')
            dummy['fvc_ano'] = ds
            variables.append('fvc_ano')

    encoding = {}

    compressor = zarr.Blosc(cname='zlib', clevel=2, shuffle=1)

    encoding = {}

    for var in variables:
        var_lower = var.lower()
        #scaling, data_min, data_max = get_scale_and_offset(var)
        #dummy[var_lower].attrs['data_min'] = data_min
        #dummy[var_lower].attrs['data_max'] = data_max
        if 'dayofyear' in dummy[var_lower].dims:
            if 'hour' in  dummy[var_lower].dims:
                chunking = tuple(CHUNKS_HOURLY_MSC.values())
            else:
                chunking = tuple(CHUNKS_MSC.values())
        else:
            if 'hour' in  dummy[var_lower].dims:
                chunking = tuple(CHUNKS_HOURLY.values())
            else:
                chunking = tuple(CHUNKS.values())

        encoding[var_lower] = {
            'compressor': compressor,
            'chunks': chunking,
            'dtype': 'float32',
            #'_FillValue': -32767,
            #'dtype': 'int16',
            #'scale_factor': scaling['scale_factor'],
            #'add_offset': scaling['add_offset']
        }

    dummy.to_zarr(out_path, compute=False, encoding=encoding, mode='w')

    for var in variables:
        add_stats_group(path=out_path, group_name=var)

def calculate_anomaly(x: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    gb = x.groupby('time.dayofyear')
    msc = gb.mean('time').compute()

    msc_0 = msc.copy().assign_coords(dayofyear=np.arange(1 - 366, 1))
    msc_1 = msc.copy().assign_coords(dayofyear=np.arange(367, 367 + 366))
    msc_stack = xr.concat((msc_0, msc, msc_1), dim='dayofyear')
    msc_smooth = msc_stack.rolling(dayofyear=17, min_periods=5, center=True).mean().sel(dayofyear=slice(1, 366))

    anomalies = gb - msc_smooth
    anomalies = anomalies.drop('dayofyear')

    return msc_smooth, anomalies


def batch_calculate_anomaly(var: str, uid: str, out_path: str, lat_subset: slice | None = None) -> None:

    dummy = xr.open_zarr(out_path)

    if lat_subset is not None:
        dummy = dummy.isel(lat=lat_subset)

    da = dummy[var.lower()]

    if 'mask' not in dummy.data_vars:
        raise KeyError(
            f'provided cube at \'{out_path}\' has no data variable \'mask\'.'
        )

    lat_chunk_bounds = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros(1, dtype=int), np.cumsum(da.chunksizes['lat']))), 2)
    lon_chunk_bounds = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros(1, dtype=int), np.cumsum(da.chunksizes['lon']))), 2)

    with tqdm(desc=f'   > Anomalies: {var.lower()}', total=len(lat_chunk_bounds) * len(lon_chunk_bounds)) as pbar:
        for lat_i, lat_chunk_bound in enumerate(lat_chunk_bounds):
            for lon_i, lon_chunk_bound in enumerate(lon_chunk_bounds):
                lat_slice = slice(*lat_chunk_bound)
                lon_slice = slice(*lon_chunk_bound)

                da_sel = da.isel(lat=lat_slice, lon=lon_slice).load()

                msc, ano = calculate_anomaly(da_sel)

                # Seasonality
                msc_da = xr.Dataset()
                msc_da[f'{var.lower()}_msc'] = msc

                if 'hour' in msc_da.dims:
                    msc_da = msc_da.drop_vars(['hour', 'dayofyear'])
                else:
                    msc_da = msc_da.drop_vars(['dayofyear'])

                with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                    lat_slice_ = slice(lat_subset.start + lat_slice.start, lat_subset.start + lat_slice.stop)
                    msc_da.to_zarr(out_path, consolidated=True, region={
                        'lat': lat_slice_,
                        'lon': lon_slice
                    })

                # Anomalies
                ano_da = xr.Dataset()
                ano_da[f'{var.lower()}_ano'] = ano

                if 'hour' in ano_da.dims:
                    ano_da = ano_da.drop_vars(['hour', 'time'])
                else:
                    ano_da = ano_da.drop_vars(['time'])

                with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                    lat_slice_ = slice(lat_subset.start + lat_slice.start, lat_subset.start + lat_slice.stop)
                    ano_da.to_zarr(out_path, consolidated=True, region={
                        'lat': lat_slice_,
                        'lon': lon_slice
                    })

                array_name = f'{uid}_{lat_i}_{lon_i}'
                add_stats(path=out_path, group_name=f'{var.lower()}_msc', array_name=array_name, array=msc)
                add_stats(path=out_path, group_name=f'{var.lower()}_ano', array_name=array_name, array=ano)

                pbar.update(1)


def add_anomalies(var: str, out_path: str, dryrun: bool = False, num_proc: int = 0) -> None:
    if dryrun:
        return

    if num_proc == 0:
        batch_calculate_anomaly(var=var, uid='0000', out_path=out_path)
        return

    # Split longitude into num_proc parts.
    dummy = xr.open_zarr(out_path)
    lat_chunks = np.concatenate((np.zeros(1, dtype=int), np.cumsum(dummy.chunksizes['lat'])))

    slices = []
    under_limit = 0
    for chunks in np.array_split(lat_chunks, num_proc):
        slices.append(slice(under_limit, chunks[-1]))
        under_limit = chunks[-1]

    #for slices_subset in [slices[0::2], slices[1::2]]:
    iter_items = [(var, f'{uid:04d}', out_path, lat_slice) for uid, lat_slice in enumerate(slices)]

    with mpl.Pool(num_proc) as pool:
        pool.starmap(batch_calculate_anomaly, iter_items)


def write_data(in_path: str, out_path: str, dryrun: bool = False) -> None:

    if dryrun:
        return

    dummy = xr.open_zarr(out_path)

    data = xr.open_dataset(in_path).pipe(rename).load()

    if 'time' not in data.dims:

        STATIC_CHUNKS = CHUNKS.copy()
        STATIC_CHUNKS.pop('time')
        compressor = zarr.Blosc(cname='zlib', clevel=2, shuffle=1)

        encoding = {}

        data = rename_to_lower(data)

        attrs = {
            'wtd': {
                'name': 'wtd',
                'long_name': 'Water dable depth',
                'units': 'm'
            },
            'maxrrootdepth': {
                'name': 'rootdepth',
                'long_name': 'Maximum root depth',
                'units': 'm'
            },
            'topidx': {
                'name': 'topidx',
                'long_name': 'Topographic water index',
                'units': '-'
            },
            'globveg3d': {
                'name': 'canopyheight',
                'long_name': 'Canopy height',
                'units': 'm'
            },
            'sndppt': {
                'name': 'sandfrac',
                'long_name': 'Sand fraction (0-100cm)',
                'units': '%'
            },
            'tc': {
                'name': 'treecover',
                'long_name': 'Percent tree cover',
                'units': '%'
            },
        }

        for k, v in attrs.items():
            if k in data.data_vars:
                data[k].attrs.update(**v)
                data = data.rename({k: v['name']})

        if 'wtd' in data.data_vars:
            data.wtd.attrs.pop('standard_name')
        elif 'sandfrac' in data.data_vars:
            #data = data.sel(layer=[1, 2, 3, 4, 5, 6]).mean('layer').compute()
            layer_content = \
                (data.isel(layer=0) + data.isel(layer=1)) / 2 * 5 + \
                (data.isel(layer=1) + data.isel(layer=2)) / 2 * 10 + \
                (data.isel(layer=2) + data.isel(layer=3)) / 2 * 15 + \
                (data.isel(layer=3) + data.isel(layer=4)) / 2 * 30 + \
                (data.isel(layer=4) + data.isel(layer=5)) / 2 * 40
            data = layer_content / 100
        elif 'tc' in data.data_vars:
            data['tc'] = data['tc'].where(data['tc'].notnull(), 0.0).compute()

        for var in data.data_vars:

            chunking = tuple(STATIC_CHUNKS.values())
            encoding[var.lower()] = {
                'compressor': compressor,
                'chunks': chunking,
                'dtype': 'float32',
            }

            data[var].attrs['mean'] = data[var].mean().compute().item()
            data[var].attrs['std'] = data[var].std().compute().item()

        data = data.transpose(*CHUNKS_ORDER, missing_dims='ignore').compute()

        data.to_zarr(out_path, mode='a', encoding=encoding)
        return

    if xr.infer_freq(data.time) == 'H':
        data = hourly2dayhour(data)
        data = data.drop(['hour', 'lat', 'lon'])
    else:
        data = data.drop(['lat', 'lon'])

    attrs = {
        'fvc': {
            'name': 'fvc',
            'long_name': 'Fractional vegetation cover',
            'units': '-'
        },
        'lst': {
            'name': 'lst',
            'long_name': 'Land surface temperature',
            'units': '°C'
        },
        'rh_cf': {
            'name': 'rh_cf',
            'long_name': 'Relative humidity',
            'units': '%'
        },
        't2m': {
            'name': 't2m',
            'long_name': 'Air temperature (2m)',
            'units': '°C'  # Conversion from Kelvin done below.
        },
        'tp': {
            'name': 'tp',
            'long_name': 'Total precipitation',
            'units': 'm'
        },
    }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        time_slice = slice(
            dummy.indexes['time'].get_loc(str(data.time[0].values)),
            dummy.indexes['time'].get_loc(str(data.time[-1].values)) + 1
        )

        for var in data.data_vars:
            data = data.rename({var: var.lower()})

        for k, v in attrs.items():
            if k in data.data_vars:
                data[k].attrs.update(**v)
                data = data.rename({k: v['name']})

        data = data.compute()

        data = data.transpose(*CHUNKS_ORDER, missing_dims='ignore').compute()

        if 't2m' in data.data_vars:
            data['t2m'] = data['t2m'] - 273.15

        # Compute mean.
        for var in data.data_vars:
            array_name = os.path.basename(data.encoding['source']).removesuffix('.nc')
            add_stats(path=out_path, group_name=var, array_name=array_name, array=data)

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            data.to_zarr(out_path, consolidated=True, region={'time': time_slice})


def merge_stats(out_path: str) -> None:
    z = zarr.open(out_path)

    for key, values in z.groups():
        if '_stats' in key:
            var_key = key.removesuffix('_stats')
            if var_key not in z.array_keys():
                raise KeyError(
                    f'no corresponding variable `{var_key}` found for stats `{key}`.'
                )

            n = 0.
            s = 0.
            s2 = 0.
            if len(list(values.arrays())):
                for arr_key, arr in values.arrays():
                    n += arr[0]
                    s += arr[1]
                    s2 += arr[2]

                mn = s / n
                sd = ((s2 / n) - (s / n) ** 2) ** 0.5

                z[var_key].attrs['mean'] = mn
                z[var_key].attrs['std'] = sd

                z.store.rmdir(key)

            else:
                raise RuntimeError(
                    f'no stats found in `{key}`.'
                )

    zarr.consolidate_metadata(out_path)

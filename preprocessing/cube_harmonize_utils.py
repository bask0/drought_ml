
import xarray as xr
import dask
import pandas as pd
import numpy as np
import zarr
from glob import glob
import warnings
from typing import Union

CHUNKS = {'time': 300, 'lat': 20, 'lon': 20}
CHUNKS_MSC = {'dayofyear': 366, 'lat': 20, 'lon': 20}
CHUNKS_HOURLY = {'time': 300, 'hour': 24, 'lat': 20, 'lon': 20}
CHUNKS_HOURLY_MSC = {'dayofyear': 366, 'hour': 24, 'lat': 20, 'lon': 20}
CHUNKS_ORDER = ('time', 'dayofyear', 'hour', 'lat', 'lon', 'layer',)


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


def rename(x: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    return x.rename({'latitude': 'lat', 'longitude': 'lon'})


def rename_to_lower(x: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
    if isinstance(x, xr.Dataset):
        for var in x.data_vars:
            x = x.rename({var: var.lower()})
    else:
        x.name = x.name.lower()

    return x


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
        var_lower = var.lower()

        ds = ds_orig.chunk().pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(
            time=times, hour=hours)[var].pipe(rename).transpose(*CHUNKS_ORDER, missing_dims='ignore')
        dummy[var_lower] = ds
        variables.append(var)

        if var_lower in ['t2m', 'tp', 'ssrd', 'rh_cf']:
            dummy[f'{var_lower}_msc'] = ds_orig.isel(time=0, drop=True).expand_dims(
                dayofyear=dayofyear, hour=hours)[var].pipe(rename).transpose(*CHUNKS_ORDER, missing_dims='ignore')
            variables.append(f'{var_lower}_msc')

        if var_lower == 'lst':
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

    for var in variables:
        var_lower = var.lower()

        if 'dayofyear' in dummy[var_lower].dims:
            if 'hour' in dummy[var_lower].dims:
                chunking = tuple(CHUNKS_HOURLY_MSC.values())
            else:
                chunking = tuple(CHUNKS_MSC.values())
        else:
            if 'hour' in dummy[var_lower].dims:
                chunking = tuple(CHUNKS_HOURLY.values())
            else:
                chunking = tuple(CHUNKS.values())

        encoding[var_lower] = {
            'compressor': compressor,
            'chunks': chunking,
            'dtype': 'float32',
        }

    dummy.to_zarr(out_path, compute=False, encoding=encoding, mode='w')


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

        if 'tp' in data.data_vars:
            data['tp'] = data['tp'] * 1000

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            data.to_zarr(out_path, consolidated=True, region={'time': time_slice})

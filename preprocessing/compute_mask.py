
from argparse import ArgumentParser
import xarray as xr
import numpy as np
import json


def get_slice_from_anchor(
        ds: xr.DataArray,
        lon: int,
        lat: int,
        num_chunks_lon: int = 5,
        num_chunks_lat: int = 3,
        chunk_size_lon: int = 20,
        chunk_size_lat: int = 20) -> dict[str, slice]:
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


def compute_mask(path):

    ds = xr.open_zarr(path)

    mask_static = (
        ds.canopyheight.notnull() *
        ds.rootdepth.notnull() *
        ds.percent_tree_cover.notnull() *
        ds.sandfrac.notnull() *
        ds.topidx.notnull() *
        ds.wtd.notnull()
    ).compute()

    mask_dynamic = (
        ds.t2m.isel(time=0, hour=0).notnull() *
        ds.tp.isel(time=0, hour=0).notnull() *
        ds.rh_cf.isel(time=0, hour=0).notnull() *
        ds.ssrd.isel(time=0, hour=0).notnull()
    ).compute()

    mask_fvc = ds.fvc.sel(time='2010').notnull().any('time').compute()

    mask_lst = ds.lst.sel(time='2010').notnull().any(('time', 'hour')).compute()

    common_mask = (mask_static * mask_dynamic * mask_fvc * mask_lst).drop(['time', 'hour']).compute()

    with open('./project/rois.json', 'r') as f:
        rois_json = json.load(f)

    rois = []

    for _, val in rois_json.items():
        rois.extend(val)

    lat_chunk = ds.fvc.encoding['preferred_chunks']['lat']
    lon_chunk = ds.fvc.encoding['preferred_chunks']['lon']

    mask = xr.Dataset()
    mask['mask'] = common_mask.chunk({'lat': lat_chunk, 'lon': lon_chunk})
    mask['fold_mask'] = xr.zeros_like(common_mask).expand_dims(fold=np.arange(6)).copy().chunk(
        {'fold': 1, 'lat': lat_chunk, 'lon': lon_chunk})

    roi_attr = {}
    for fold in range(6):
        fold_rois = [get_slice_from_anchor(common_mask, *roi) for roi in rois[fold::6]]
        for r, fold_roi in enumerate(fold_rois):
            mask['fold_mask'].loc[{**fold_roi, 'fold': fold}] = True
            roi_attr.update(
                {
                    f'roi_{fold}_{r}': [
                        [fold_roi['lon'].start, fold_roi['lon'].stop],
                        [fold_roi['lat'].start, fold_roi['lat'].stop]
                    ]
                }
            )

    mask['fold_mask'] = mask['fold_mask'] * mask['mask']
    mask['fold_mask'].attrs = roi_attr

    encoding = {}

    for var, chunking in zip(
            ['mask', 'fold_mask'], [(lat_chunk, lon_chunk), (1, lat_chunk, lon_chunk)]):
        encoding[var] = {
            'chunks': chunking,
        }

    mask.to_zarr(path, mode='a')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--path',
        '-p',
        type=str,
        default='/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr/')

    args = parser.parse_args()

    compute_mask(args.path)


from argparse import ArgumentParser
import xarray as xr
import numpy as np
from scipy import ndimage


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


# def get_fold_mask(
#         mask: xr.DataArray,
#         lat_grid_size: int,
#         lon_grid_size: int,
#         lat_chunk_size: int = 20,
#         lon_chunk_size: int = 20):
#     lat_grid_size = lat_grid_size * lat_chunk_size
#     lon_grid_size = lon_grid_size * lon_chunk_size
#     num_lat_rep = int(np.ceil(len(mask.lat) / lat_grid_size / 2))
#     num_lon_rep = int(np.ceil(len(mask.lon) / lon_grid_size / 2))

#     re = np.r_[num_lon_rep * [0, 1]]  # even-numbered rows
#     ro = np.r_[num_lon_rep * [2, 3]]  # odd-numbered rows
#     checkboard = np.row_stack(num_lat_rep * (re, ro))
#     checkboard = checkboard.repeat(
#         lat_grid_size, axis=0).repeat(lon_grid_size, axis=1)[:len(mask.lat), :len(mask.lon)] + 1

#     mask = mask.where(mask.lon > -25, False)
#     mask = mask.where((mask.lat <= 16.5) | (mask.lat > 32.5), False)
#     mask = mask.where((mask.lat < 60) | (mask.lon > 0), False)

#     blocks = xr.zeros_like(mask, dtype=int)
#     blocks.values = checkboard
#     blocks = blocks.where(mask, False)

#     return blocks

def get_fold_mask(
        mask: xr.DataArray,
        num_folds: int = 12,
        lat_chunk_size: int = 20,
        lon_chunk_size: int = 20):

    blocks = mask.copy().astype(int)

    num_lat_rep = int(np.ceil(len(mask.lat) / lat_chunk_size))
    num_lon_rep = int(np.ceil(len(mask.lon) / lon_chunk_size))

    for i in range(-num_lat_rep, num_lon_rep + 1):
        diag = np.eye(num_lat_rep, num_lon_rep, i, dtype=bool).repeat(
            lat_chunk_size, axis=0).repeat(lon_chunk_size, axis=1)
        blocks.values[diag] = (i % num_folds) + 1

    blocks = blocks.where(mask, 0)
    blocks = blocks.where(mask.lon > -25, 0)
    blocks = blocks.where((mask.lat <= 16.5) | (mask.lat > 32.5), 0)
    blocks = blocks.where((mask.lat < 60) | (mask.lon > 0), 0)

    return blocks.compute()


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

    common_mask.values[:] = ndimage.binary_opening(common_mask.values, iterations=2)

    lat_chunk = ds.fvc.encoding['preferred_chunks']['lat']
    lon_chunk = ds.fvc.encoding['preferred_chunks']['lon']

    blocks = get_fold_mask(
        mask=common_mask,
        lat_chunk_size=lat_chunk,
        lon_chunk_size=lon_chunk
    )

    mask = xr.Dataset()
    mask['mask'] = common_mask.chunk({'lat': lat_chunk, 'lon': lon_chunk})
    mask['fold_mask'] = blocks.chunk({'lat': lat_chunk, 'lon': lon_chunk})

    encoding = {}

    for var in ['mask', 'blocks']:
        encoding[var] = {
            'chunks': (lat_chunk, lon_chunk,),
        }

    mask.to_zarr(path, mode='a', consolidated=True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--path',
        '-p',
        type=str,
        default='/Net/Groups/BGI/scratch/bkraft/drought_data/cube.zarr/')

    args = parser.parse_args()

    compute_mask(args.path)


import xarray as xr


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

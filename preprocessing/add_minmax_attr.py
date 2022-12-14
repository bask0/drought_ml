
import argparse
import os
import netCDF4 as nc
import xarray as xr

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Add min and max to .nc file.')

    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='path to a netcdf file `file.nc`, `file.max.nc` and `file.min.nc` must be present, required'
    )
    parser.add_argument(
        '--varname', '-n',
        type=str,
        required=True,
        help='variable name, must be present in `file.nc`, `file.max.nc` and `file.min.nc`, required'
    )
    parser.add_argument(
        '--clean_up',
        action='store_true',
        help='if flag is set, the `file.max.nc` and `file.min.nc` are removed.'
    )

    args = parser.parse_args()

    file: str = args.path

    var: str = args.varname
    clean_up: bool = args.clean_up

    if not os.path.isfile(file):
        raise ValueError(
            f'the file \'{file}\' does not exist.'
        )

    ds = nc.Dataset(file, 'a', format='NETCDF4')
    variables = ds.variables.keys()

    if var not in variables:
        var_old = var
        var = var.upper()
    if var not in variables:
        raise KeyError(
            f'variable {var_old}/{var} not found in dataset with variables {" ,".join(variables)}.'
        )

    ds_xr = xr.open_dataset(file)[var]
    q = ds_xr.quantile([0.0, 1.0]).compute()
    ds_xr.close()

    min_val = q.isel(quantile=0).item()
    max_val = q.isel(quantile=1).item()

    ds.setncattr('min_val', min_val)
    ds.setncattr('max_val', max_val)
    ds.close()

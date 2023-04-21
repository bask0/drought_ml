import xarray as xr
import matplotlib.pyplot as plt
import os
from glob import glob
from multiprocessing import Pool
import tempfile
import shutil


import dask
dask.config.set(scheduler='synchronous')


def plot_t(ds_t, vmin, vmax, title, save_dir=None, dpi=200, cmap='BrBG', rotate=0, globe=True, t=False, is_hourly=True, **kwargs):

    extends = []

    if isinstance(vmin, str):
        if isinstance(vmax, str):
            extends.append('both')
        else:
            extends.append('min')
    else:
        if isinstance(vmax, str):
            extends.append('max')
        else:
            extends.append('neither')


    if is_hourly:
        time = ds_t.time.dt.strftime("%Y-%m-%d-%H").item()
    else:
        time = ds_t.time.dt.strftime("%Y-%m-%d").item()        

    fig, ax = plot_map(
        ds_t.load(),
        label=title,
        title=f'{time}',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

        #cbar.ax.xaxis.set_tick_params(color='w')
        #cbar.outline.set_edgecolor('w')
        #plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

        #ax.set_title('')
        #ax.text(0.3, 0.3, f'{title} – {time}', transform=ax.transAxes, color='w')

    # fig.tight_layout()

    path = os.path.join(save_dir, f'frame_{time}.png')

    fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    del ds_t, ax, title

def animate(
        ds, vmin, vmax, title, save_path, timerange=-1, levels=None,
        globe=True, rotate_speed=0, dpi=150, cmap='BrBG', fps=15, ncpus=1, use_pil=False,
        png_out=False):

    tmpdir = tempfile.mkdtemp(dir=os.path.dirname(save_path))

    if isinstance(ds, xr.Dataset):
        raise TypeError('must pass an xr.DataArray, not an xr.Dataset.')
    elif isinstance(ds, xr.DataArray):
        if not isinstance(vmin, (int, float, str)):
            raise TypeError('`vmin` must be numeric or a string.')
        if not isinstance(vmax, (int, float, str)):
            raise TypeError('`vmax` must be numeric or a string.')
        if not isinstance(title, str):
            raise TypeError('`title` must be a string.')
        if not isinstance(cmap, str):
            raise TypeError('`cmap` must be a string.')

        ntime = len(ds.time)

    else:
        if not isinstance(vmin, list):
            raise TypeError('`vmin` must be a list as multiple datasets were passed.')
        if not isinstance(vmax, list):
            raise TypeError('`vmax` must be a list as multiple datasets were passed.')
        if not isinstance(title, list):
            raise TypeError('`title` must be a list as multiple datasets were passed.')
        if not isinstance(cmap, list):
            raise TypeError('`cmap` must be a list as multiple datasets were passed.')

        ntime = len(ds[0].time)

    if timerange == -1:
        timerange = range(ntime)
    elif isinstance(timerange, int):
        timerange = range(timerange)

    if isinstance(ds, list):
        timedim = ds[0]['time']
    else:
        timedim = ds['time']

    is_hourly = xr.infer_freq(timedim) == 'H'

    try:

        par_kwargs = [{
            'ds_t': ds.isel(time=t) if isinstance(ds, xr.DataArray) else [ds[i].isel(time=t) for i in range(len(ds))],
            'vmin': vmin,
            'vmax': vmax,
            'title': title,
            'save_dir': tmpdir,
            'dpi': dpi,
            'cmap': cmap,
            'rotate': t * rotate_speed,
            'globe': globe,
            't': t,
            'is_hourly': is_hourly
        } for t in timerange]

        par_args = [list(el.values()) for el in par_kwargs]

        # parcall(plot_t, ds_args, dry_run=False, num_cpus=ncpus, vmin=vmin, vmax=vmax, levels=levels,
        #     globe=globe, save_dir=tmpdir, title=title, cmap=cmap, dpi=dpi);

        with Pool(processes=min(len(par_args), ncpus)) as pool:
            pool.starmap(plot_t, par_args)

        fp_in = os.path.join(tmpdir, 'frame*.png')
        paths = sorted(glob(fp_in))

        if not png_out:
            #command = f'convert -delay {100 / fps} -loop 0 {fp_in} {save_path}'
            command = f'ffmpeg -y -threads 16 -framerate {fps} -pattern_type glob -i \'{fp_in}\' -b:v 0  -crf 40 -c:v libvpx-vp9 -pix_fmt yuva420p {save_path}/{title.replace(" ", "_").lower()}.webm'
            print(command)
            os.system(command)
            print('Done.')

        else:
            if os.path.isdir(save_path):
                shutil.rmtree(save_path)
            os.rename(tmpdir, save_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

        


import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeat

import numpy as np


def beautify_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.patch.set_facecolor('None')
    ax.set_xlabel('')

    return ax


def savefig(fig, path, dpi=300, transparent=False, **kwargs):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=transparent, **kwargs)


def subplots_ortho(ncols=1, center=(0, 0), figsize=(8, 8), glob=True):
    map_proj = ccrs.Orthographic(*center)
    map_proj._threshold /= 100.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={'projection': map_proj},
        squeeze=False,
        gridspec_kw={'wspace': 0})

    for ax in axes.flat:
        # generate a basemap with country borders, oceans and coastlines
        ax.add_feature(cfeat.LAND, color='0.6')
        ax.add_feature(cfeat.OCEAN, color='0.4')
        ax.add_feature(cfeat.COASTLINE, lw=.3, zorder=10)
        ax.add_feature(cfeat.BORDERS, linestyle='dotted', lw=0.3)
        if glob:
            ax.set_global()
    return fig, axes[0, :]


def subplots_ortho_dense(text=None, add_borders=True):
    fig = plt.figure(constrained_layout=True, figsize=(4, 5.4))

    map_proj = ccrs.Orthographic(0, 0)
    map_proj._threshold /= 100.

    gs = fig.add_gridspec(11, 5, hspace=0)
    ax1 = fig.add_subplot(gs[:4, :], projection=map_proj, frameon=False)
    ax2 = fig.add_subplot(gs[4:, :], projection=map_proj, frameon=False)

    for ax in [ax1, ax2]:
        # generate a basemap with country borders, oceans and coastlines
        ax.add_feature(cfeat.LAND, color='0.8')
        #ax.add_feature(cfeat.OCEAN, color='1.0')
        ax.add_feature(cfeat.COASTLINE, lw=.5, zorder=10)
        if add_borders:
            ax.add_feature(cfeat.BORDERS, lw=0.3)

    ax1.set_extent([-10, 60, 32, 70])
    ax2.set_extent([-18, 50, -31, 14.5])

    kwargs = dict(
        color='k',
        transform=ccrs.Geodetic(),
        ha='left',
        va='center',
        fontsize=9,
        path_effects=[pe.withStroke(linewidth=1, foreground='w')], zorder=9999
    )

    if text is not None:
        ax2.text(-14, -10, text, **kwargs)

    return fig, [ax1, ax2]


def map_hist(ds, ax, cmap, vmin=None, vmax=None, robust=True, label=None, histogram_placement=[0.04, 0.35, 0.4, 0.3], bins=30, add_contour=True, contour_lw=0.4, **kwargs):

    axh = ax.inset_axes(histogram_placement)

    weights = np.cos(np.deg2rad(ds.lat))

    cmap = plt.get_cmap(cmap)

    data_median = ds.weighted(weights).quantile(0.5).item()
    data_mean = ds.weighted(weights).mean().item()

    weights = weights.expand_dims(lon=ds.lon, axis=1).values

    mask = np.isfinite(ds.values)
    weights = weights[mask]
    ds_data = ds.values[mask]
    
    if vmin is None or vmax is None:
        if robust:
            q = 0.02
        else:
            q = 0.0

        xmin, xmax = np.quantile(ds_data, [q, 1.0-q])

    if vmin is None:
        vmin = xmin
    if vmax is None:
        vmax = xmax

    n, bns, patches = axh.hist(ds_data, bins=np.linspace(vmin, vmax, bins), weights=weights, **kwargs)
    if add_contour:
        axh.hist(ds_data, histtype='step', color='k', bins=bns, lw=contour_lw, weights=weights)

    bin_centers = 0.5 * (bns[:-1] + bns[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))

    # axh.hist(ds, weights=cs, bins=bins, color='0.5', range=(vmin, vmax))

    axh.set_title('')
    axh.spines['right'].set_visible(False)
    axh.spines['left'].set_visible(False)
    axh.spines['top'].set_visible(False)
    axh.set_yticks([])
    axh.patch.set_facecolor('None')
    axh.set_xlabel('')

    #min_tick = (vmax - vmin) / (bins - 1) / 2
    #max_tick = vmax - min_tick

    # axh.set_xticks(np.linspace(bin_centers[0], bin_centers[-1], 3))
    # labs = np.linspace(vmin, vmax, 3)
    # labs_new = []
    # for l in labs:
    #     if l % 1 == 0:
    #         labs_new.append(int(l))
    #     else:
    #         labs_new.append(l)

    #axh.set_xticklabels(labs_new)
    axh.xaxis.set_tick_params(labelsize=8, pad=0.2)
    axh.axvline(data_mean, color='0.3', lw=1.2)
    axh.axvline(data_median, color='0.3', ls=':', lw=1.2)

    if label is not None:
        axh.set_xlabel(label, fontsize=9)

    axh.text(
        0.02, 0.2,
        'Area-weighted distribution\nof map values with mean ({0})\nand median ({1}).'.format(u'\u2500', u'\u2509'),
        ha='left',
        va='top',
        transform=ax.transAxes,
        fontsize=8,
        style='italic'
    )

    
def plot_map(ds, label, title=None, vmin=None, vmax=None, robust=True, cmap='coolwarm', do_center=False, add_hist=True):

    fig, axes = subplots_ortho_dense()

    plot_kwargs = {}

    if do_center:
        plot_kwargs.update(center=0)


    ds.plot(ax=axes[0], transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap, robust=robust, vmin=vmin, vmax=vmax, **plot_kwargs)
    ds.plot(ax=axes[1], transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap, robust=robust, vmin=vmin, vmax=vmax, **plot_kwargs)

    for ax in axes:
        ax.set_title('')

    if title is not None:
        axes[1].text(
            0.22, 0.74,
            title,
            ha='center',
            va='top',
            transform=ax.transAxes,
            fontsize=8
        )

    if add_hist:
        map_hist(ds, ax=axes[1], cmap=cmap, label=label, vmin=vmin, vmax=vmax, robust=robust)

    gridline_spec = dict(
        linewidth=0.7,
        color='k',
        alpha=0.7,
        linestyle='--'
    )

    for lon in [0, 30, 60]:
        axes[0].plot([lon, lon], [-30, 80], transform=ccrs.PlateCarree(), **gridline_spec)

    for lat in [45, 60]:
        axes[0].plot([-12, 90], [lat, lat], transform=ccrs.PlateCarree(), **gridline_spec)

    for lon in [0, 30, 60]:
        lat_min = -40 if lon == 30 else 5
        axes[1].plot([lon, lon], [lat_min, 80], transform=ccrs.PlateCarree(), **gridline_spec)

    for lat, (lon_min, lon_max) in zip([-30, -15, 0, 15], [(14, 35), (10, 55), (8, 46), (-30, 55)]):
        axes[1].plot([lon_min, lon_max], [lat, lat], transform=ccrs.PlateCarree(), **gridline_spec)


    return fig, axes

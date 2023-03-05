
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeat


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

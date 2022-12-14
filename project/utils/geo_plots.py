
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


def subplots_ortho_dense():
    fig = plt.figure(constrained_layout=True, figsize=(4, 5.5))

    map_proj = ccrs.Orthographic(0, 0)
    map_proj._threshold /= 100.

    gs = fig.add_gridspec(11, 5, hspace=0)
    ax1 = fig.add_subplot(gs[:4, :], projection=map_proj, frameon=False)
    ax2 = fig.add_subplot(gs[4:, :], projection=map_proj, frameon=False)
    ax3 = fig.add_axes([0.01, 0.01, 0.4, 0.4], projection=map_proj, frameon=False)

    for ax in [ax1, ax2, ax3]:
        # generate a basemap with country borders, oceans and coastlines
        ax.add_feature(cfeat.LAND, color='0.6')
        # ax.add_feature(cfeat.OCEAN, color='1.0')
        ax.add_feature(cfeat.COASTLINE, lw=.5, zorder=10)
        ax.add_feature(cfeat.BORDERS, lw=0.3)

    ax1.set_extent([-10, 60, 30, 70])
    ax2.set_extent([-18, 50, -31, 18])
    ax3.set_extent([-70, -41, -36, 0])

    kwargs = dict(
        color='k', transform=ccrs.Geodetic(), ha='center', va='center', fontsize=13,
        path_effects=[pe.withStroke(linewidth=1, foreground='w')], zorder=9999
    )
    ax1.text(11, 39, 'Eurasia', **kwargs)
    ax2.text(20, 0, 'Africa', **kwargs)
    ax3.text(-56, -5, 'South\nAmerica', **kwargs)

    return fig, [ax1, ax2, ax3]

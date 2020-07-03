# Copyright 2020 Tom Eulenfeld, MIT license

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import obspy
from obspy import UTCDateTime as UTC
from obspy.imaging.beachball import beach

from load_data import get_events
from util.events import events2lists, load_webnet
from util.imaging import add_ticklabels, convert_coords2km


def get_bounds():
    bounds=['2018-05-10', '2018-05-11 03:10:00', '2018-05-12 06:00', '2018-05-21', '2018-05-25', '2018-06-19']
    return [UTC(b) for b in bounds]


def get_colors():
    cmap1 = plt.get_cmap('Accent')
    N = 8
    colors = [cmap1((i+0.5) / N) for i in range(N)]
    colors = ['#7fc97f', '#beaed4', '#fdc086', '#ff9896', '#386cb0']
    return colors


def get_cmap(extend=False):
    bounds = [b.matplotlib_date for b in get_bounds()]
    if extend:
        bounds[-1]=UTC('2018-05-26').matplotlib_date
    colors = get_colors()[:len(bounds)-1]
    cmap = ListedColormap(colors, name='AccentL')
    norm = BoundaryNorm(bounds, ncolors=len(colors))
    return cmap, norm


def plot_events_stations_map_depth(events, inv=None, figsize=(8,8), out=None, show=True,
                                   dpi=300, convert_coords=False, plotall=True, colorbar=True, label=True, ms=9):
    id_, time, lon, lat, dep, mag, *_ = zip(*events2lists(events.filter('magnitude > 1.8')))
    _, _, lon2, lat2, dep2, _, *_ = zip(*events2lists(events))
    print(time[0], time[-1])

    fig = plt.figure(figsize=figsize)
    ax3 = fig.add_axes((0.1, 0.5, 0.4, 0.4))
    ax4 = fig.add_axes((0.52, 0.5, 0.35, 0.4), sharey=ax3)
    ax5 = fig.add_axes((0.1, 0.5-0.37, 0.4, 0.35), sharex=ax3)
    def _on_lims_changed(ax, boo=[True]):
        if boo[0]:
            boo[0] = False
            if ax == ax5:
                ax4.set_xlim(ax5.get_ylim()[::-1])
            if ax == ax4:
                ax5.set_ylim(ax4.get_xlim()[::-1])
            boo[0] = True

    mpl = [t.matplotlib_date for t in time]
    ax5.invert_yaxis()
    ax5.callbacks.connect('ylim_changed', _on_lims_changed)
    ax4.callbacks.connect('xlim_changed', _on_lims_changed)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position("top")
    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position("top")

    if convert_coords:
        latlon0 = None if convert_coords==True else convert_coords
        x, y = zip(*convert_coords2km(list(zip(lat, lon)), latlon0=latlon0))
        x2, y2 = zip(*convert_coords2km(list(zip(lat2, lon2)), latlon0=latlon0))
    else:
        x, y = lon, lat
        x2, y2 = lon2, lat2
    mag = np.array(mag)

    cmap, norm = get_cmap()
    if plotall:
        ax3.scatter(x2, y2, 4, color='0.6')
        ax4.scatter(dep2, y2, 4, color='0.6')
        ax5.scatter(x2, dep2, 4, color='0.6')
    ax3.scatter(x, y, ms, mpl, cmap=cmap, norm=norm)
    ax4.scatter(dep, y, ms, mpl, cmap=cmap, norm=norm)
    ax5.scatter(x, dep, ms, mpl, cmap=cmap, norm=norm)
    if colorbar:
        ax7 = fig.add_axes([0.56, 0.42, 0.34, 0.02])
        cmap, norm = get_cmap(extend=True)
        cbar = ColorbarBase(ax7, cmap=cmap, norm=norm, orientation='horizontal', format=DateFormatter('%Y-%m-%d'), extend='max', spacing='proportional')#, extendfrac=0.2)
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels()[:-1] + ['until 2018-06-19'], rotation=60, ha='right', rotation_mode='anchor')
        xticks = cbar.ax.get_xticks()
        xticks = np.mean([xticks[1:], xticks[:-1]], axis=0)
        for xpos, label in zip(xticks, 'abcde'):
            # this line only works for matplotlib 3.1 at the moment
            cbar.ax.annotate(label, (xpos, 0.1), ha='center', fontstyle='italic')

    ax4.set_xlabel('depth (km)')
    ax5.set_ylabel('depth (km)')
    if convert_coords:
        if label:
            ax3.set_xlabel('easting (km)')
            ax3.set_ylabel('northing (km)')
            ax5.set_xlabel('easting (km)')
            ax4.set_ylabel('northing (km)')
    else:
        if label:
            ax3.set_xlabel('longitude')
            ax3.set_ylabel('latitude')
            ax5.set_xlabel('longitude')
            ax4.set_ylabel('latitude')
    if out:
        plt.savefig(out, dpi=dpi)
    if show:
        plt.show()
    return fig


def plot_events2018(events, bb=None):
    bbfname = 'data/focal_mechanism_2018swarm.txt'
    fig = plot_events_stations_map_depth(events, convert_coords=LATLON0, show=False)
    bbs = np.genfromtxt(bbfname, names=True)
    xys = [(0, 3), (0, 4), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0),
           (1, 2), (0, 5), (0, 6), (1, 3), (1, 5), (1, 4)]
    for i, bb in enumerate(bbs):
        _, lat, lon, dep, mag, *sdr = bb
        xy = convert_coords2km([(lat, lon)], latlon0=LATLON0)[0]
        ax = fig.axes[0]
        xy2= (xys[i][0]*3.8-1.9, -(xys[i][1] + xys[i][0]*0.5)/6*4 + 2.2)
        ax.plot(*list(zip(xy, xy2)), lw=0.5, color='k', zorder=10)
        b = beach(sdr, xy=xy2, width=0.5, linewidth=0.5, facecolor='C0')
        ax.add_collection(b)
    fig.axes[0].set_xlim(-2.5, 2.5)
    fig.axes[0].set_ylim(-2.3, 2.7)
    fig.axes[1].set_xlim(5.95, 10.95)
    fig.savefig('figs/eventmap.pdf', bbox_inches='tight', pad_inches=0.1)


def plot_events2018_interpretation(events):
    fig = plot_events_stations_map_depth(events, show=False, plotall=False, colorbar=False, label=False, ms=4)
#    eqs = load_webnet(stations=False)
#    color = 'gray'
#    color = '#9bd7ff'
#    fig.axes[0].scatter(eqs.lon.values, eqs.lat.values, 4, eqs.time.values, marker='o', zorder=-1)
#    fig.axes[1].scatter(eqs.dep.values, eqs.lat.values, 4, eqs.time.values, marker='o', zorder=-1)
#    fig.axes[2].scatter(eqs.lon.values, eqs.dep.values, 4, eqs.time.values, marker='o', zorder=-1)
#    fig.axes[0].set_xlim(-2.5, 2.5)
#    fig.axes[0].set_ylim(-2.3, 2.7)
#    lon, lat, dep = 12.42396, 50.22187, 8.5
#    lon, lat, dep = 12.45993, 50.27813, 8.5
#    lon, lat, dep = 12.45993, 50.22187, 11.5
    lon, lat, dep = 12.45993, 50.22187, 8.5
    kw = dict(zorder=-1, vmin=3, vmax=4, cmap='Greys', edgecolors='face', linewidths=0.01)
#    kw = dict(zorder=-1, vmin=5, vmax=6.5, cmap='Greys')
#    kw = dict(zorder=-1, vmin=1.55, vmax=1.75, cmap='Greys')
#    load_mousavi()
    try:
        plot_mousavi(2, dep, 0, 1, 4, ax=fig.axes[0], **kw)
        plot_mousavi(0, lon, 2, 1, 4, ax=fig.axes[1], **kw)
        im = plot_mousavi(1, lat, 0, 2, 4, ax=fig.axes[2], **kw)
    except OSError:
        import traceback
        print(traceback.format_exc())
        print('Mousavi model is not provided in this repository')
        im = None
    kw = dict(color='red', lw=2, zorder=0)
    kw = dict(color='C1', alpha=0.8, lw=2, zorder=0)
    ax0, ax1, ax2, *_ = fig.axes
    ax0.axhline(lat, **kw)
    ax0.axvline(lon, **kw)
    ax1.axvline(dep, **kw)
    ax2.axhline(dep, **kw)
    ax0.set_xlim(12.388, 12.53187)
    ax0.set_ylim(50.10935, 50.3344)
    ax1.set_xlim(0, 12)
    ax1.set_xticks(ax2.get_yticks())
    cax2 = fig.add_axes([0.56, 0.42, 0.34, 0.02])
    cmap, norm = get_cmap()
    cbar = ColorbarBase(cax2, cmap=cmap, norm=norm, orientation='horizontal',
                        format=DateFormatter('%Y-%m-%d'))
    cax2.set_xticklabels([])
    cax2.set_xlabel('intra-cluster\nS wave velocity (km/s)', labelpad=12)
    for i, label in enumerate('abcde'):
        cax2.annotate(label, ((i+0.5) / 5, 0.1), xycoords='axes fraction',
                      ha='center', fontstyle='italic')
    for i, label in enumerate([4.16, 3.49, 3.66, 3.85, 3.72]):
        cax2.annotate(label, ((i+0.5) / 5, -1.2), xycoords='axes fraction',
                      ha='center')
    lonticks = [12.40, 12.45, 12.50]
    latticks = [50.15, 50.20, 50.25, 50.30]
    add_ticklabels(ax0, lonticks, latticks, fmt='%.2f°')
    if im is not None:
        cax = fig.add_axes([0.56, 0.27, 0.34, 0.02])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        for i, label in enumerate([4.16, 3.49, 3.66, 3.85, 3.72]):
            cax.annotate('', (label - 3, 1.1), (label - 3, 2.5), xycoords='axes fraction',
                         textcoords='axes fraction',
                         arrowprops=dict(width=1, headwidth=4, headlength=6, color=cmap(i)))#, xycoords='axes fraction',
        cax.set_xlabel('S wave velocity (km/s)\nMousavi et al. (2015)')
    fig.savefig('figs/eventmap_interpretation.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_mousavis():
    deps = [2.5, 5.5, 8.5, 11.5]
    for d in deps:
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        kw = dict(zorder=-1, vmin=3, vmax=4, cmap='Greys', edgecolors='face', linewidths=0.01)
#        kw = dict(zorder=-1, vmin=5, vmax=6.5, cmap='Greys')
        kw = dict(zorder=-1, vmin=1.55, vmax=1.75, cmap='Greys')
        im = plot_mousavi(2, d, 0, 1, 5, ax=ax, **kw)
        ax.set_xlim(12.388, 12.53187)
        ax.set_ylim(50.10935, 50.3344)
        lonticks = [12.40, 12.45, 12.50]
        latticks = [50.15, 50.20, 50.25, 50.30]
        add_ticklabels(ax, lonticks, latticks, fmt='%.2f°')
        fig.colorbar(im, ax=ax)
        fig.savefig('tmp/mousavi_vpvs_depth_%04.1fkm.png' % d, bbox_inches='tight', pad_inches=0.1)


def events2txt(events, fname):
    id_, time, lon, lat, dep, mag, *_ = zip(*events2lists(events))
    np.savetxt(fname, np.transpose([lat, lon, mag]), fmt=['%.6f', '%.6f', '%.1f'], header='lat lon mag')


def plot_topomap(events=None):
    from cartopy.feature import NaturalEarthFeature as NEF
    from matplotlib.colors import LinearSegmentedColormap
    import shapely.geometry as sgeom
    from util.imaging import EA_EURO, GEO, add_scale, de_border, add_ticklabels, plot_elevation

    mlf = [(12.410, 50.261), (12.485, 50.183), (12.523, 50.127),
           (12.517, 50.125), (12.538, 50.131), (12.534, 50.130),
           (12.543, 50.109), (12.547, 50.083), (12.547, 50.074),
           (12.545, 50.066), (12.546, 50.057), (12.576, 50.034),
           (12.594, 50.016), (12.632, 50.004), (12.665, 49.980)]
    fig = plt.figure()
#    ax = fig.add_axes([0, 0, 1, 1], projection=EA_EURO)
    ax = fig.add_subplot(111, projection=EA_EURO)
    extent = [12.05, 12.85, 50, 50.45]
    ax.set_extent(extent, crs=GEO)

    # Create an inset GeoAxes showing the location of the Solomon Islands.
    box = ax.get_position().bounds
    subax = fig.add_axes([box[0]+box[2]-0.23, box[1]+box[3]-0.3, 0.28, 0.28], projection=EA_EURO)
    subax.set_extent([8, 16, 47, 55], GEO)
    subax.add_feature(NEF('physical', 'land', '10m'), facecolor='0.7', alpha=0.5, rasterized=True) #facecolor='sandybrown'
    subax.add_feature(NEF('physical', 'coastline', '10m'), facecolor='none', edgecolor='k', linewidth=0.5, rasterized=True)
    subax.add_feature(NEF('cultural', 'admin_0_boundary_lines_land', '10m'), facecolor='none', edgecolor='k', linewidth=0.5, rasterized=True)
    subax.add_geometries([sgeom.box(extent[0], extent[2], extent[1], extent[3])], GEO,
                          facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
    lonticks = [12.2, 12.4, 12.6, 12.8]
    latticks = [50, 50.1, 50.2, 50.3, 50.4]
    add_ticklabels(ax, lonticks, latticks)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plot_elevation(ax, shading=False, cmap=LinearSegmentedColormap.from_list('littlegray', ['white', '0.5']),
                   azimuth=315, altitude=60, rasterized=True)
    add_scale(ax, 10, (12.15, 50.02))
    de_border(ax, edgecolor='0.5', rasterized=True)
    eqs, sta = load_webnet(stations=True)
    if eqs is not None:
        ax.scatter(eqs.lon.values, eqs.lat.values, 4, '#9bd7ff', alpha=0.4, marker='.', transform=GEO, rasterized=True)
    if events is not None:
        _, _, lon, lat, _, mag, *_ = zip(*events2lists(events))
        ax.scatter(lon, lat, 4, 'C0', marker='o', transform=GEO)
    ax.plot(*list(zip(*mlf)), color='0.5', transform=GEO)
    ax.annotate('MLF', (12.505, 50.145), None, GEO._as_mpl_transform(ax), size='x-small', zorder=10, rotation=290)
    used_stations = 'NKC LBC VAC KVC STC POC SKC KRC ZHC'.split()
    sta = sta[sta.station.isin(used_stations)]
    ax.scatter(sta.lon.values, sta.lat.values, 100, marker='^', color='none', edgecolors='k', transform=GEO, zorder=10)
    for idx, s in sta.iterrows():
        xy = (2, 2) if s.station not in ('KOPD', 'KAC') else (-10, 5)
        ax.annotate(s.station, (s.lon, s.lat), xy, GEO._as_mpl_transform(ax), 'offset points', size='x-small', zorder=10)
    x0, y0 = EA_EURO.transform_point(LATLON0[1], LATLON0[0], GEO)
    ax.add_geometries([sgeom.box(x0-2500, y0-2300, x0+2500, y0+2700)], EA_EURO,
                       facecolor='none', edgecolor='C1', linewidth=2, alpha=0.8, zorder=11)
    fig.savefig('figs/topomap.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    return sta


def load_mousavi_numpy():
    fname = 'data/mousavi2015.txt'
    mod = np.loadtxt(fname)
    return mod


def load_mousavi(*args):
    mod = load_mousavi_numpy()
    if len(args) == 0:
        print('Arguments: ind1, indv, ind2, ind3, ind4')
        print('Choose 4 indices from 0-5 (lon, lat, dep, vp, vs, vp/vs)')
        print('ind1: select value of this index (0-3)')
        print('ind2: x axis (0-3)')
        print('ind3: y axis (0-3)')
        print('ind4: values (3-5)')
        print('indv: Choose val from')
        lons = np.array(sorted(set(mod[:, 0])))
        lats = np.array(sorted(set(mod[:, 1])))
        deps = np.array(sorted(set(mod[:, 2])))
        print(lons)
        print(lats)
        print(deps)
        print('Choose extract_ind 3-5 (vp, vs, vp/vs)')
        return lons, lats, deps, mod
    ind1, indv, ind2, ind3, ind4 = args
    import pandas
    df = pandas.DataFrame(mod)
    df2 = df[df.iloc[:, ind1] == indv].pivot(ind3, ind2, ind4)
    return df2.columns.to_numpy(), df2.index.to_numpy(), df2.to_numpy()


def get_corners(x, y, z):
    x2 = 0.5 * (x[1:] + x[:-1])
    y2 = 0.5 * (y[1:] + y[:-1])
    return x2, y2, z[1:-1, 1:-1]


def plot_mousavi(*args, ax=None, **kw):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    nd1, indv, ind2, ind3, ind4 = args
    x, y, z = load_mousavi(*args)
    x, y, z = get_corners(x, y, z)
    x, y = np.meshgrid(x, y)
    print('value range', np.min(z), np.max(z))
    im = ax.pcolormesh(x, y, z, **kw)
    if ind3 == 2:
        ax.set_ylim(12, 0)
    if ind3 == 1:
        ax.set_ylim(49.8, 50.8)
    if ind2 == 1:
        ax.set_xlim(49.8, 50.8)
    if ind2 == 0:
        ax.set_xlim(12.1, 12.65)
    return im


MAG = 1.5

TXTFILE = 'tmp/cat_2018_mag>1.8.txt'
LATLON0 = (50.25, 12.45)

if __name__ == '__main__':
#    plot_mousavis()
    print('load events')
    allevents = get_events()
    print('finished loading')
    allevents = obspy.Catalog(sorted(allevents, key=lambda ev: ev.origins[0].time))
    events = allevents.filter(f'magnitude > {MAG}')
    events2txt(events, TXTFILE)
    plot_events2018(allevents)
    sta = plot_topomap(events=allevents)
    plot_events2018_interpretation(allevents)
    plt.show()

# Copyright 2020 Tom Eulenfeld, MIT license

import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.dates import DateFormatter
import numpy as np
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.api as sm

from plot_maps import get_bounds, get_cmap, get_colors
from traveltime import repack, filters, LATLON0
from util.imaging import convert_coords2km


def fit_velocity(dist, tmax1, plot=True):
#    rlm = RLM(dist/1000 , np.abs(tmax1))
#    rlm = RLM(np.abs(tmax1), dist/1000, M=sm.robust.norms.Hampel(1, 2, 4))
    s = dist / 1000
    t = np.abs(tmax1)
    o = np.ones(len(t))
#    rlm = RLM(d/t, np.ones(len(d)), M=sm.robust.norms.Hampel(1, 2, 4))
    rlm = RLM(s/t, o, M=sm.robust.norms.Hampel(1, 2, 4))
    res = rlm.fit(maxiter=100)
    v = res.params[0]
    w = res.weights
#    scale = res.scale
#    v = np.median(s/t)
    from statsmodels.robust.scale import mad
    scale = mad(s/t, center=v, c=1)
    tmax = np.max(s) / v
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.scatter(tmax1, dist)
        ax.plot((-tmax, 0, tmax), (np.max(dist), 0, np.max(dist)))
        ax2 = fig.add_subplot(122)
        ax2.hist(dist/1000/np.abs(tmax1), bins=np.linspace(3, 5, 21))
    return v, w, scale


def stats_vs_time(dist, tmax1, tmax2, evids, times1, times2, bounds):
    ress = []
    r = []
    for t1, t2 in zip(bounds[:-1], bounds[1:]):
        print(t1, t2)
        cond = np.logical_and.reduce((times1 <= t2, times1 > t1, times2 <= t2, times2 > t1, tmax1 != 0))
        dist_, tmax1_, tmax2_, t1_ = filters(cond, dist, tmax1, tmax2, times1)
        ind = tmax2_!=0
        tmax_=tmax1_
        v, weights, scale = fit_velocity(dist_, tmax_, plot=False)
        r.append((t1, v, weights, scale, dist_, tmax_))
        ress.append('{!s:10} {:.3f} {} {:.3f}'.format(t1, np.round(v, 3), len(dist_), np.round(scale, 3)))

    fig = plt.figure(figsize=(10, 8.5))
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(333, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(334, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(335, sharex=ax1, sharey=ax1)
    ax6 = fig.add_subplot(336)
    axes = [ax1, ax2, ax3, ax4, ax5]
    colors = get_colors()
    for i, (t1, v, weights, scale, dist, tmax1) in enumerate(r):
        tmax = np.max(dist) / 1000 / v
        c = colors[i]
        ax = axes[i]
        cmap = LinearSegmentedColormap.from_list('w_%d' % i, ['white', c])
        ax.scatter(tmax1, dist, c=weights, cmap=cmap, edgecolors='k')
        ax.plot((-tmax, 0, tmax), (np.max(dist), 0, np.max(dist)), color=c)
        ax.annotate('N=%d' % len(dist), (0.03, 0.03), xycoords='axes fraction')
        ax.annotate(r'$v_{{\rm{{S}}}}{{=}}({:.2f}{{\pm}}{:.2f})$km/s'.format(v, scale), (0.97, 0.03), xycoords='axes fraction', ha='right')

    bins=np.linspace(2, 6, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    heights = np.diff(bins)
    for i, (t1, v, weights, scale, dist, tmax1) in enumerate(r):
        c = colors[i]
        vmedian = np.median(dist/1000/np.abs(tmax1))
        data = np.histogram(dist/1000/np.abs(tmax1), bins=bins)[0]
        data = data / np.max(data)
        ax6.barh(centers, data, height=heights, left=i-0.5*data, color=c, alpha=0.5)
        ax6.errorbar(i, v, scale, fmt='o', color=c)
        ind = np.digitize(vmedian, bins) - 1
        ax6.plot([i, i-0.5*data[ind]], [vmedian, vmedian], ':', color=c)
        ax6.plot([i, i+0.5*data[ind]], [vmedian, vmedian], ':', color=c)

    for ax, label in zip(axes + [ax6], 'abcdef'):
        ax.annotate(label + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top')
    ax1.set_ylabel('event distance (m)')
    ax4.set_ylabel('event distance (m)')
    ax4.set_xlabel('lag time (s)')
    ax5.set_xlabel('lag time (s)')
    ax6.set_ylabel('apparent S wave velocity (km/s)')
    ax6.set_xlim(-0.5, 4.5)
    ax6.set_ylim(2.5, 5)
    ax6.set_xticks([])
    fig.tight_layout()
    bounds = ax6.get_position().bounds
    ax7 = fig.add_axes([bounds[0], bounds[1]-0.02, bounds[2], 0.01])
    cmap, norm = get_cmap()
    cbar = ColorbarBase(ax7, cmap=cmap, norm=norm, orientation='horizontal', format=DateFormatter('%Y-%m-%d'))
    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    for i, label in enumerate('abcde'):
        ax6.annotate(label, ((i+0.5) / 5, 0.04), xycoords='axes fraction', ha='center', fontstyle='italic')
    fig.savefig('figs/vs_time.pdf', bbox_inches='tight', pad_inches=0.1)
    return ress


PKL_FILE = 'tmp/stuff_onlymax.pkl'
NSR = 0.1

with open(PKL_FILE, 'rb') as f:
    stuff = pickle.load(f)

dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), evids, azi, inc, (coords1, coords2), v, (times1, times2), phase1 = repack(stuff.values())
coords1 = convert_coords2km(coords1, LATLON0)
coords2 = convert_coords2km(coords2, LATLON0)
x1, y1, z1 = [np.array(bla) for bla in zip(*coords1)]
x2, y2, z2 = [np.array(bla) for bla in zip(*coords1)]
cond1 = np.logical_or(np.logical_and(azi>0, azi<300), inc < 20, inc>50)

cond = np.logical_and.reduce((snr2 <= NSR, dist > 200, cond1, phase1>0.00))
dist, tmax1, tmax2, max1, max2, snr1, snr2, snr3, snr4, evids, coords1, coords2, v, times1, times2, phase1 = filters(cond, dist, tmax1, tmax2, max1, max2, snr1, snr2, snr3, snr4, evids, np.array(coords1), np.array(coords2), v, times1, times2, phase1)
resdepth = stats_vs_time(dist, tmax1, tmax2, evids, times1, times2, get_bounds())
print('\n'.join(resdepth))
plt.show()
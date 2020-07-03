# Copyright 2020 Tom Eulenfeld, MIT license

import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import scipy

from load_data import get_events
from plot_maps import get_bounds, get_colors
from util.events import events2lists


def odr_brute(x, y, norm=None, **kw):
    x = np.array(x)
    y = np.array(y)
    def error(g):
        dist = np.abs(g[0] * x - y) / (g[0] ** 2 + 1) ** 0.5
        if norm == 'mad':
            return np.median(np.abs(dist))
        elif norm == 'lms':
            return (np.median(dist ** 2)) ** 0.5
        return np.linalg.norm(dist, norm) / len(x)
    return scipy.optimize.brute(error, **kw)


def standard_error_of_estimator(y1, y2, x):
    return (np.sum((y1-y2)**2) / np.sum((x-np.mean(x))**2) / len(y1)) ** 0.5


def PSpicks(picks_, ot):
    picks = picks_
    stations = [p[0].split('.')[1] for p in picks]
    stations = sorted({sta for sta in stations if stations.count(sta) == 2})
    picks2 = {'P': [], 'S': []}
    for seed, phase, time in sorted(picks):
        sta = seed.split('.')[1]
        if sta in stations:
            picks2[phase[0]].append(time - ot)
    if not len(stations) == len(picks2['P']) == len(picks2['S']):
        print(len(stations), len(picks2['P']), len(picks2['S']))
        raise RuntimeError()
    return stations, np.array(picks2['P']), np.array(picks2['S'])


def cond1(x, y):
    return np.abs(y - x * 1.7) < 0.1

def cond2(x, y):
    dist = (x ** 2 + (y/vpvs0) ** 2) ** 0.5
    return dist < 0.35

def double_diff(picks, apply_cond1=True, apply_cond2=True):
    """
    Lin and Shearer 2007

    picks: list of [stations, P picks, S picks] (in seconds)
    """
    x = []
    y = []
    for picks1, picks2 in list(combinations(picks, 2)):
        sta1, p1, s1 = picks1
        sta2, p2, s2 = picks2
        sta = set(sta1) & set(sta2)
        index1 = [i for i, s in enumerate(sta1) if s in sta]
        index2 = [i for i, s in enumerate(sta2) if s in sta]
        p1 = p1[index1]
        p2 = p2[index2]
        s1 = s1[index1]
        s2 = s2[index2]
        dp = p2 - p1
        ds = s2 - s1
        if apply_cond1:
            x1 = dp - np.median(dp)
            y1 = ds - np.median(ds)
            good = cond1(x1, y1)
            dp = dp[good]
            ds = ds[good]
            if len(dp) == 0:
                continue
        if apply_cond2:
            x1 = dp - np.median(dp)
            y1 = ds - np.median(ds)
            good = cond2(x1, y1)
            dp = dp[good]
            ds = ds[good]
            if len(dp) == 0:
                continue
        x1 = dp - np.mean(dp)
        y1 = ds - np.mean(ds)
        x.extend(list(x1))
        y.extend(list(y1))
    x = np.array(x)
    y = np.array(y)
    return x, y

def odr_fit(x, y, scale=1.7, lim=(1.5, 1.9), Ns=101, norm=1):
    b3, _, vpvsgrid, err = odr_brute(x, y/scale,
                                     ranges=[[lim[0]/scale, lim[1] / scale]],
                                     Ns=Ns, norm=norm, full_output=True)
    b3 = b3[0]*scale
    vpvsgrid = vpvsgrid * scale
    return b3, vpvsgrid, err

plt.rc('font', size=20)
plt.rc('mathtext', fontset='cm')

events = get_events().filter('magnitude > 1.0')
events.events = sorted(events, key=lambda e: e.origins[0].time)

bounds = get_bounds()
fig, axs = plt.subplots(2, 5, figsize=(20, 8.5), sharex='row', sharey='row')
colors = get_colors()

for i, (t1, t2, vpvs0) in enumerate(
        zip(bounds[:-1], bounds[1:], [1.68, 1.66, 1.69, 1.68, 1.69])):
    ax = axs[0, i]
    events2 = events.filter(f'time > {t1}', f'time < {t2}')
    events2 = events2lists(events2)
    picks = {}
    ids = list(zip(*events2))[0]
    for id_, ot, lat, lon, dep, mag, picks_ in events2:
        picks[id_] = PSpicks(picks_, ot)

    picks = list(picks.values())
    x, y = double_diff(picks)
#    xrem, yrem = double_diff(picks, apply_cond1=True, apply_cond2=False)
#    ax.scatter(xrem, yrem, 16, marker='.', c='0.7', rasterized=True)

    m1 = np.nanmedian(np.array(y) / np.array(x))
    os = 0 #0.2 * i
    b2, vpvsgrid, err = odr_fit(x, y, scale=1, lim=[1.48, 1.92], Ns=111, norm=1)
    b3, vpvsgrid, err = odr_fit(x, y, scale=vpvs0, lim=[1.48, 1.92], Ns=221, norm=1)
    b4 = odr_brute(x, y, ranges=[(1.5, 1.9)], Ns=101, norm=2)[0]
    b5 = odr_brute(x, y, ranges=[(1.5, 1.9)], Ns=101, norm='lms')[0]
    print('error', standard_error_of_estimator(y, b3*x, x))
    print(t1, t2,  '{:.2f}  {:.2f}  {:.2f}  {:.2f}'.format(m1, b3, b2, b4))
    x1 = np.min(x)
    x2 = np.max(x)
    c = colors[i]
    ax.scatter(x, y, 16, marker='.', c=c, rasterized=True)

    ax.plot((x1, x2), (b3 * x1+os, b3 * x2+os), '-k')
    ax.plot((x1, x2), (1.5 * x1+os, 1.5 * x2+os), 'k', ls=(0, (1, 2)))
    ax.plot((x1, x2), (1.9 * x1+os, 1.9 * x2+os), 'k', ls=(0, (1, 2)))
    ax.annotate('N=%s\n$\\mathdefault{v_{\\rm{P}}/v_{\\rm{S}}{=}%.2f}$' % (len(events2), b3) , (0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom')
    ax.annotate('abcde'[i] + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top')
    ax = axs[1, i]
    ax.axvline(b3, color='k', zorder=-1)
    ax.plot(vpvsgrid, 1000 * err, color=c, lw=4)
    ax.annotate('fghij'[i] + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top')

for i in range(1, 5):
    plt.setp(axs[0, i].get_yticklabels(), visible=False)

axs[0, 0].set_ylabel('differential S wave\ntravel time $\\hat{\\delta}t_{\\rm S}^{ij}$ (s)')
axs[0, 2].set_xlabel(r'differential P wave travel time $\hat{\delta}t_{\rm P}^{ij}$ (s)')
axs[0, 0].set_xticks([-0.2, 0, 0.2])
axs[1, 0].set_xticks([1.5, 1.6, 1.7, 1.8, 1.9])
axs[1, 0].set_xlim(1.48, 1.92)
axs[1, 0].set_yticks([7, 8, 9, 10, 11, 12])

axs[1, 0].set_ylabel('mean absolute error (ms)')
axs[1, 2].set_xlabel(r'velocity ratio $\mathdefault{v_{\rm P}/v_{\rm S}}$')

fig.tight_layout(w_pad=-2)
fig.savefig('figs/vpvs.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
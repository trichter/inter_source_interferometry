# Copyright 2020 Tom Eulenfeld, MIT license
"""
extract travel times (stuff) and create different plots
"""


from copy import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.imaging.beachball import beach
import scipy

from load_data import LATLON0
from util.imaging import convert_coords2km
from util.signal import trim2
from util.source import plot_farfield, nice_points, plot_pnt
from util.xcorr2 import plot_corr_vs_dist_hist, plot_corr_vs_dist, velocity_line
from xcorr import add_distbin


def filters(ind, *args):
    return [arg[ind] for arg in args]


def repack(obj):
    jaja = [np.array(bla) if len(bla) not in (2, 4) else bla for bla in zip(*obj)]
    jaja[5] = [np.array(bla) for bla in zip(*jaja[5])]
    jaja[9] = [np.array(bla) for bla in zip(*jaja[9])]
    jaja[-2] = [np.array(bla) for bla in zip(*jaja[-2])]
    return jaja


def get_stuff(stream, stream_phase=None, v=3.6, mindist=0, only_maxima=False):
    if stream_phase is None:
        stream_phase = [None] * len(stream)
    stuff = {}
    for tr, trphase in zip(stream, stream_phase):#[:200]:
        assert tr.stats.stack.group == trphase.stats.stack.group
        dist = tr.stats.dist
        if dist < mindist:
            continue
        d = tr.data
        absd = np.abs(d)
        ad = d if only_maxima else absd
        dt = tr.stats.delta
        starttime = tr.stats.starttime
        midsecs = (tr.stats.endtime - starttime) / 2

        ind_max1 = np.argmax(ad)
        max1 = d[ind_max1]
        phase1 = 0 if trphase is None else trphase.data[ind_max1]
        tmax1 = dt * ind_max1 - midsecs
        ind_maxima = scipy.signal.argrelmax(d)[0]
        ind_maxima = ind_maxima[ind_maxima != ind_max1]
        ind_maxima = sorted(ind_maxima, key=lambda i: absd[i], reverse=True)
        # inverse of signal to noise ratio
        snr1 = 0
        snr2 = absd[ind_maxima[NOISE_IND]] / abs(max1)
#        snr3 = ad[ind_maxima[19]] / abs(max1)
        snr3 = 0
        snr4 = 0
#        snr5 = 0
        max2 = 0
        tmax2 = 0
#        for i in ind_maxima:
#            dift = abs(i - ind_max1) * dt
#            if dift < 0.1:
#                # side lobe
#                snr4 = max(snr4, ad[i] / abs(max1))
#                continue
#            if abs(dift - 2 * dist / 1000 / v) < 0.05 and max2 == 0 and snr1 == 0 and np.sign(tmax1) != np.sign(dt * i - midsecs):
#                # second maxima
#                max2 = d[i]
#                tmax2 = dt * i - midsecs
#            else:
#                snr1 = max(snr1, ad[i] / abs(max1))
        if True:
            if tmax1 > 0:
                ad[len(ad)//2:] = 0
            else:
                ad[:len(ad)//2] = 0
            i = np.argmax(ad)
            dift = abs(i - ind_max1) * dt
#            if abs(dift - 2 * dist / 1000 / v) < 0.05:
            if abs(dift - 2 * dist / 1000 / v) < 0.1:
                max2 = d[i]
                tmax2 = dt * i - midsecs



        v2 = dist / 1000 / abs(tmax1)
        # distance, seconds 1, seconds2 (or None), maximum value 1 (+ or -), max value 2 (+ or -),
        # noise levels:
        #  1: largest peak, not side lobe and not max2 / maximum
        #  2: 10th local maximum / maximum
        #  3: 20th local maximum / maximum
        #  4: side lobe of maximum / maximum
        # event id, azimuth, inclination
        # coordinates event1 and event2, apparent velocity
        bla = (dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), tr.stats.stack.group,
               tr.stats.azi, tr.stats.inc,
               (_coords(tr.stats, '1'), _coords(tr.stats, '2')), v2, (tr.stats.event_time1, tr.stats.event_time2),
               phase1)
        stuff[tr.stats.stack.group] = bla
    return stuff


def plot_stuff_map_depth(stuff, dist=None, tmax1=None, max1=None, max2=None, snr2=None, coords1=None, coords2=None, v=None,
                         figsize=(8,8), out=None, show=True,
                         dpi=300, convert_coords=True, plot_lines=False):
    if dist is None:
        dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), evids, azi, inc, (coords1, coords2), v, *_ = repack(stuff.values())
        cond = np.logical_and.reduce((snr2 <= 0.25, dist > 200, inc<50, inc>40, azi>340))
        dist, tmax1, max1, max2, snr2, coords1, coords2, v = filters(cond, dist, tmax1, max1, max2, snr2, coords1, coords2, v)
        print(len(dist))
    if convert_coords:
        coords1 = convert_coords2km(coords1, latlon0=LATLON0)
        coords2 = convert_coords2km(coords2, latlon0=LATLON0)
    x1, y1, dep1 = zip(*coords1)
    x2, y2, dep2 = zip(*coords2)
    x, y, dep = zip(*np.mean([coords1, coords2], axis=0))

#        import utm
#        x, y = zip(*[utm.from_latlon(lat_, lon_)[:2]
#                     for lat_, lon_ in zip(lat, lon)])
#        x1, y1 = zip(*[utm.from_latlon(lat_, lon_)[:2]
#                     for lat_, lon_ in zip(lat1, lon1)])
#        x2, y2 = zip(*[utm.from_latlon(lat_, lon_)[:2]
#                     for lat_, lon_ in zip(lat2, lon2)])
#
#        x1 = (np.array(x1) - np.mean(x)) / 1000
#        y1 = (np.array(y1) - np.mean(y)) / 1000
#        x2 = (np.array(x2) - np.mean(x)) / 1000
#        y2 = (np.array(y2) - np.mean(y)) / 1000
#        x = (np.array(x) - np.mean(x)) / 1000
#        y = (np.array(y) - np.mean(y)) / 1000
#    else:
#        x, y = lon, lat


    fig = plt.figure(figsize=figsize)
    ax3 = fig.add_axes((0.1, 0.5, 0.4, 0.4))
    ax4 = fig.add_axes((0.52, 0.5, 0.35, 0.4), sharey=ax3)
    ax5 = fig.add_axes((0.1, 0.5-0.37, 0.4, 0.35), sharex=ax3)
    if plot_lines:
        ax6 = fig.add_axes((0.52, 0.5-0.37, 0.35, 0.35))
    else:
        ax6 = fig.add_axes((0.6, 0.5-0.37, 0.02, 0.35))
    def _on_lims_changed(ax, boo=[True]):
        if boo[0]:
            boo[0] = False
            if ax == ax5:
                ax4.set_xlim(ax5.get_ylim()[::-1])
            if ax == ax4:
                ax5.set_ylim(ax4.get_xlim()[::-1])
            boo[0] = True

    ax5.invert_yaxis()
    ax5.callbacks.connect('ylim_changed', _on_lims_changed)
    ax4.callbacks.connect('xlim_changed', _on_lims_changed)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position("top")
    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position("top")


    if plot_lines:
        cmap = plt.get_cmap()
        colors = cmap((np.array(v) - 3.2) / 1)
        colors = cmap(np.sign(max1))
#        for xx1, yy1, dp1, xx2, yy2, dp2, cc in zip(x1, y1, dep1, x2, y2, dep2, colors):
#            ax3.plot((xx1, xx2), (yy1, yy2), color=cc)
#            ax4.plot((dp1, dp2), (yy1, yy2), color=cc)
#            ax5.plot((xx1, xx2), (dp1, dp2), color=cc)
        import matplotlib as mpl
        def _rs(a, b, c, d):
            return list(zip(list(zip(a, b)), list(zip(c, d))))
        ln_coll1 = mpl.collections.LineCollection(_rs(x1, y1, x2, y2), colors=colors)
        ln_coll2 = mpl.collections.LineCollection(_rs(dep1, y1, dep2, y2), colors=colors)
        ln_coll3 = mpl.collections.LineCollection(_rs(x1, dep1, x2, dep2), colors=colors)
        ax3.add_collection(ln_coll1)
        ax4.add_collection(ln_coll2)
        ax5.add_collection(ln_coll3)
        ax3.set_xlim(np.min(x), np.max(x))
        ax3.set_ylim(np.min(y), np.max(y))
        ax4.set_xlim(np.min(dep), np.max(dep))
        im = ax6.scatter(tmax1, dist, 10, np.sign(max1))
        plt.colorbar(im, ax=ax6)
    else:
        vmax = None
        vmax=None
        v = np.sign(max1)
        ax3.scatter(x, y, 10, v, vmax=vmax)
        ax4.scatter(dep, y, 10, v, vmax=vmax)
        sc = ax5.scatter(x, dep, 10, v, vmax=vmax)
        plt.colorbar(sc, cax=ax6)
#    cbar.ax.invert_yaxis()
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-2.3, 2.7)
    ax4.set_xlim(5.95, 10.95)
    ax4.set_xlabel('depth (km)')
    ax5.set_ylabel('depth (km)')
    if convert_coords:
        ax3.set_xlabel('EW (km)')
        ax3.set_ylabel('NS (km)')
        ax5.set_xlabel('EW (km)')
        ax4.set_ylabel('NS (km)')
    else:
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax5.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
    if out:
        plt.savefig(out, dpi=dpi)
    if show:
        plt.show()


def plot_corr_vs_dist2times():
    plt.rc('font', size=12)
    stream1 = read(NAME, 'H5')
    stream2 = read(NAMED, 'H5')
#    from xcorr import add_distbin
#    add_distbin(ccs_stack, 1, 21)
    stream1.stack('{distbin}').sort(['distbin']).normalize()
    stream2.stack('{distbin}').sort(['distbin']).normalize()
    for tr in stream1:
        tr.stats.dist = tr.stats.distbin
    for tr in stream2:
        tr.stats.dist = tr.stats.distbin

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

    for j, stream in enumerate([stream1, stream2]):
        max_ = max(stream.max())
        max_dist = max(tr.stats.dist for tr in stream)
        ax = [ax1, ax2][j]
        for i, tr in enumerate(stream):
            starttime = tr.stats.starttime
            mid = starttime + (tr.stats.endtime - starttime) / 2
            t = tr.times(reftime=mid)
            scaled_data = tr.stats.dist + tr.data * max_dist / max_ / 50
            ax.plot(t, scaled_data, 'k', lw=1)
        velocity_line(3.6, ax, t, [max_dist], lw=2)
        ax.set_xlabel('lag time (s)')
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, None)
        ax.legend(loc='lower right')
        label = ['a)  coda window with normalization', 'b)  direct S wave window'][j]
        ax.annotate(label, (0.03, 0.985), xycoords='axes fraction', va='top', size='medium')
    ax1.set_ylabel('event distance (m)')
    fig.savefig(f'{OUT2}corrs_vs_dist.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.rcdefaults()


def plot_corr(stream, annotate=True, expr='{station}.{channel} {evid1}-{evid2}',
              expr2='{evid1}-{evid2} {dist:.1f}m',
              figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    max_ = max(stream.max())
    stream = stream.copy()
    trim2(stream, -0.5, 0.5, 'mid')
    stream.traces.append(copy(stream).stack()[0])
    for i, tr in enumerate(stream):
        starttime = tr.stats.starttime
        mid = starttime + (tr.stats.endtime - starttime) / 2
        t = tr.times(reftime=mid)
        if i == len(stream) - 1:
            max_ = np.max(np.abs(tr.data))
        ax.plot(t, i + tr.data/max_*1.5, 'k' if i < len(stream)-1 else 'C1', lw=1 if i < len(stream)-1 else 2)
        if annotate:
            try:
                label = expr.format(**tr.stats)
                ax.annotate(label, (t[-1], i), (-5, 0),
                            'data', 'offset points',
                            ha='right')
            except KeyError:
                pass
    dist = stream[0].stats.dist
    tt = dist / 4000
    ax.axvline(0, color='0.7')
    ax.axvline(-tt, color='C1', alpha=0.5)
    ax.axvline(tt, color='C1', alpha=0.5)
    if expr2 is not None:
        ax.annotate(expr2.format(**stream[0].stats), (t[0]+0.05, len(stream)), (5, 0),
                    'data', 'offset points', size='x-large')


def make_crazy_plot(stuff, figsize=None):
    dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), evids, azi, inc, *_ = repack(stuff.values())

    cond = np.logical_and(snr2 <= NSR, dist > 200)
    tmax1, max1, max2, azi, inc, snr2 = filters(cond, tmax1, max1, max2, azi, inc, snr2)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(513, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(514, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(515, sharex=ax1, sharey=ax1)
#    ax3 = fig.add_subplot(133)
#    ax1.plot(azis, vals, 'x')
#    ax2.plot(incs, vals, 'x')
#    vmax = max(abs(min(vals)), max(vals))
#    vmax = 5
#    s = 10*(snrs)**2
#    ax1.scatter(azis[snrs<SNR], incs[snrs<SNR], marker='.', color='0.75')
#    ax1.scatter(bazs[snrs<SNR], np.mod(180-incs[snrs<SNR], 180), marker='.', color='0.75')
    val = (np.abs(max1) - np.abs(max2)) / np.abs(max1) * np.sign(tmax1)
    sc = ax1.scatter(azi, inc, 10, c=val, cmap='coolwarm')
#    sc = ax1.scatter(bazs[snrs>=SNR], np.mod(180-incs[snrs>=SNR], 180), s, c=-vals[snrs>SNR], vmin=-vmax, vmax=vmax, cmap='coolwarm')
    cbar = plt.colorbar(sc, ax=ax1)
    ankw = dict(xy=(0.05, 0.98), xycoords='axes fraction')
    ax1.annotate('relation of maxima', **ankw)
    sc = ax2.scatter(azi, inc, c=snr2, marker='.', cmap='viridis_r')
#    sc = ax2.scatter(bazs, np.mod(180-incs, 180), c=snrs, marker='.', vmax=vmax, cmap='viridis_r')
    cbar = plt.colorbar(sc, ax=ax2)
    ax2.annotate('noise level of larger maxima', **ankw)
    sc = ax3.scatter(azi[max2 != 0], inc[max2 != 0], c=np.abs(snr2*max1/max2)[max2 != 0], marker='.', cmap='viridis_r')
#    sc = ax3.scatter(bazs[snrs>=SNR], np.mod(180-incs[snrs>=SNR], 180), c=snrs2[snrs>=SNR], marker='.', vmax=vmax, cmap='viridis_r')
    cbar = plt.colorbar(sc, ax=ax3)
    ax3.annotate('noise level of smaller maxima', **ankw)
    sc = ax4.scatter(azi, inc, c=np.log10(np.abs(max1)), marker='.', cmap='viridis_r')
#    sc = ax4.scatter(bazs[snrs>=SNR], np.mod(180-incs[snrs>=SNR], 180), marker='.', vmax=vmax, c=snrs1[snrs>=SNR], cmap='viridis_r')
    cbar = plt.colorbar(sc, ax=ax4)
    ax4.annotate('log10(abs(max1))', **ankw)

    sc = ax5.scatter(azi, inc, c=np.sign(max1), marker='.', cmap='coolwarm')
#    sc = ax4.scatter(bazs[snrs>=SNR], np.mod(180-incs[snrs>=SNR], 180), marker='.', vmax=vmax, c=snrs1[snrs>=SNR], cmap='viridis_r')
    cbar = plt.colorbar(sc, ax=ax5)
    ax5.annotate('sign(max1)', **ankw)
    ax1.set_ylabel('inc (deg)')
    ax2.set_ylabel('inc (deg)')
    ax3.set_ylabel('inc (deg)')
    ax4.set_ylabel('inc (deg)')
    ax5.set_ylabel('inc (deg)')
#    ax1.set_xlabel('azi (deg)')
#    ax2.set_xlabel('azi (deg)')
#    ax3.set_xlabel('azi (deg)')
#    ax4.set_xlabel('azi (deg)')
    ax5.set_xlabel('azi (deg)')
    return True


def _coords(stats, no):
    return np.array([stats.get('elat' + no), stats.get('elon' + no), stats.get('edep' + no)])


def plot_maxima_vs_dist(stuff, v=None, figsize=None):
    plt.rc('font', size=12)
    dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), *_ = repack(stuff.values())
    print('above noise:', np.count_nonzero(snr2<=NSR))
    print('below noise:', np.count_nonzero(snr2>NSR))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    s = 9
#    plt.scatter(*filters(np.logical_and(max1<0, snr2>0.25), tmax1, dist), s=s, edgecolors='gray', facecolors='none')
    ax.scatter(*filters(np.logical_and(max1>0, snr2<=NSR), tmax1, dist), s=s, c='C0', label='positive polarity', zorder=5)  # #5f7ee8
    ax.scatter(*filters(np.logical_and(max1<0, snr2<=NSR), tmax1, dist), s=s, c='#df634e', label='negative polarity', zorder=4)
    ax.scatter(*filters(snr2>NSR, tmax1, dist), s=s, c='0.7', label='low SNR', zorder=3)
    d, t1, t2, s2 = filters(np.logical_and(0<=snr2*max1/max2, snr2*max1/max2<=NSR), dist, tmax1, tmax2, snr2)
#    print(len(d))
#    plt.plot([t1, t2], [d, d], color='0.7', zorder=-10)
#    plt.plot(0.5 * (t1 + t2), d, '|', color='0.7', zorder=-10)
#    d, t1, t2, s2 = filters(np.logical_and(-0.25<=snr2*max1/max2, snr2*max1/max2<=0), dist, tmax1, tmax2, snr2)
#    print(len(d))
#    plt.plot([t1, t2], [d, d], color='C9', zorder=-10)
#    plt.plot(0.5 * (t1 + t2), d, '|', color='C9', zorder=-10)
    if v is not None:
        velocity_line(v, ax, lw=2, zorder=6, color='C5')
    ax.axhline(200, 0.2, 0.8, color='C5', ls='--')
    ax.set_xlabel('lag time (s)')
    ax.set_ylabel('event distance (m)')
    ax.legend()
    plt.rcdefaults()


def print_evpairs(stuff, v=None):
    dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), evids, *_ = repack(stuff.values())
    print('two peaks, same sign')
    cond = np.logical_and(0<=snr2*max1/max2, snr2*max1/max2<=NSR)
    print(list(filters(cond, evids)[0]))
    print('two peaks, different sign')
    cond = np.logical_and(-NSR<=snr2*max1/max2, snr2*max1/max2<=0)
    print(list(filters(cond, evids)[0]))
    print('peaks not fitting to velocity')
    cond = np.logical_and(snr2 <= NSR, np.abs(dist / 1000 / v - np.abs(tmax1)) > 0.1)
    print(list(filters(cond, evids)[0]))
    cond2 = snr2 <= NSR
    evids1 = list(filters(cond, evids)[0])
    evids2 = list(filters(cond2, evids)[0])
    count(evids1, evids2)


def plot_maxima_vs_dist_event(stack, evid, v=None, figsize=None):
    points = []
    points2 = []
    lines = []
    for tr in stack:#[:200]:
        starttime = tr.stats.starttime
        mid = starttime + (tr.stats.endtime - starttime) / 2
        dt = tr.stats.delta
        d1 = tr.slice(None, mid).data[::-1]
        d2 = tr.slice(mid, None).data
        arg1 = np.argmax(d1)
        arg2 = np.argmax(d2)
        m1 = d1[arg1]
        m2 = d2[arg2]
        mr1 = np.median(d1[scipy.signal.argrelmax(d1)])
        mr2 = np.median(d2[scipy.signal.argrelmax(d2)])
        m3 = max(m1, m2)
        c = 'r' if tr.stats.stack.group.startswith(evid) else 'b'
        if m1 / m3 > 0.8 and m2/m3 > 0.8:
            points.append((arg1 * dt, m1, m1 / mr1, tr.stats.dist, c))
            points.append((-arg2 * dt, m2, m2 / mr2, tr.stats.dist, c))
        elif m1 / m3 > 0.8:
            points.append((arg1 * dt, m1, m1 / mr1, tr.stats.dist, c))
        elif m2 / m3 > 0.8:
            points.append((-arg2 * dt, m2, m2 / mr2, tr.stats.dist, c))
    plt.figure(figsize=figsize)
    t, m_, rel, dist, color = zip(*points)
    size = [10 if r > 10 else 3 for r in rel]
    plt.scatter(t, dist, s=size, c=color)
    if v is not None:
        velocity_line(v, plt.gca(), lw=2)
    print(len(points), len(points2), len(lines))
    return stuff


def analyze_stuff(stuff):
    # outdated
    from collections import Counter
    strange_events = []
    normal_events = []
    for k, v in stuff.items():
        dist, t1, m1, mr1, t2, m2, mr2 = v
        t = t1 if m1 > m2 else t2
        if not -0.1 < dist / 4000 - t < 0.1:
            strange_events.extend(k.split('-'))
        else:
            normal_events.extend(k.split('-'))

    return Counter(strange_events), Counter(normal_events)


def count(evids1, evids2):
    from collections import Counter
    bla1 = Counter([x for evid in evids1 for x in evid.split('-')])
    print(bla1)
    bla2 = Counter([x for evid in evids2 for x in evid.split('-')])
    x = {key: round(val / bla2[key], 3) for key, val in bla1.items()}
    print(dict(sorted(x.items(), key=lambda k: k[1], reverse=True)))

def plot_hists(stuff, figsize=None):
    dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), evids, azi, inc, *_, phase1 = repack(stuff.values())
    cond = np.logical_and(snr2 <= NSR, dist > 200)
    dist, tmax1, tmax2, max1, max2, snr1, snr2, snr3, snr4, azi, inc, phase1 = filters(cond, dist, tmax1, tmax2, max1, max2, snr1, snr2, snr3, snr4, azi, inc, phase1)
    N = 5
    M = 5
    ms = 3
    plt.figure(figsize=figsize)
    plt.subplot(M, N, 1)
    plt.hist(max1, 1001)
    plt.xlabel('max1')
    plt.subplot(M, N, 2)
    plt.scatter(dist, max1, s=ms)
    plt.subplot(M, N, 3)
    plt.hist(max2[max2!=0], 1001)
    plt.xlabel('max2')
    plt.subplot(M, N, 4)
    plt.scatter(dist[max2!=0], max2[max2!=0], s=ms)
    plt.subplot(M, N, 5)
    plt.hist(snr1, 101)
    plt.xlabel('snr1')
    plt.subplot(M, N, 6)
    plt.scatter(dist, snr1, s=ms)
    plt.subplot(M, N, 7)
    plt.hist(snr2, 101)
    plt.xlabel('snr2')
    plt.subplot(M, N, 8)
    plt.scatter(dist, snr2, s=ms)
    plt.xlabel('versus dist')
    plt.subplot(M, N, 9)
    plt.hist(snr4, 101)
    plt.xlabel('snr4')
    plt.subplot(M, N, 10)
    plt.scatter(dist, snr4, s=ms)

    plt.subplot(M, N, 11)
    plt.scatter(max1, snr4, s=ms)
    plt.xlabel('max1')
    plt.ylabel('snr4')
    plt.subplot(M, N, 12)
    plt.scatter(max1, max2, s=ms)
    plt.xlabel('max1')
    plt.ylabel('max2')

    plt.subplot(M, N, 13)
    plt.scatter(max1, snr1, s=ms)
    plt.xlabel('max1')
    plt.ylabel('snr1')
    plt.subplot(M, N, 14)
    plt.scatter(max1, snr2, s=ms)
    plt.xlabel('max1')
    plt.ylabel('snr2')

    plt.subplot(M, N, 15)
    plt.scatter(inc, max1, s=ms)
    plt.xlabel('inc')
    plt.ylabel('max1')

    plt.subplot(M, N, 16)
    plt.scatter(inc, snr1, s=ms)
    plt.xlabel('inc')
    plt.ylabel('snr1')

    plt.subplot(M, N, 17)
    plt.scatter(dist, max1, s=ms)
    plt.xlabel('dist')
    plt.ylabel('max1')

    plt.subplot(M, N, 18)
    plt.scatter(dist, snr1, s=ms)
    plt.xlabel('dist')
    plt.ylabel('snr1')

    val = (np.abs(max1) - np.abs(max2)) / np.abs(max1) * np.sign(tmax1)
    plt.subplot(M, N, 19)
    histlist = [inc[val>0], inc[val<=0]]
    plt.hist(histlist, 21, stacked=True, label=['right', 'left'])
    plt.xlabel('inc')
    plt.legend()


    plt.subplot(M, N, 20)
    histlist = [azi[max1>0], azi[max1<=0]]
    plt.hist(histlist, 21, stacked=True, label=['+', '-'])
    plt.xlabel('azi')
    plt.ylabel('sign (+-)')
    plt.legend()

    plt.subplot(M, N, 21)
    plt.scatter(snr2, phase1, s=ms)
    plt.xlabel('snr2')
    plt.ylabel('phase1')
    plt.subplot(M, N, 22)
    plt.scatter(max1, phase1, s=ms)
    plt.xlabel('max1')
    plt.ylabel('phase1')
    plt.subplot(M, N, 23)
    plt.hist(phase1, 1001)
    plt.xlabel('phase1')

    plt.rc('font', size=12)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 90, 19)
    histlist = [inc[val>0], inc[val<=0]]
    kw = dict(edgecolor='C0', linewidth=0.01)

    _, _, ps = ax.hist(histlist, bins, stacked=True, label=['right', 'left'], **kw)
    for p in ps[0]:
        p.set_zorder(11)
    for p in ps[1]:
        p.set_zorder(10)
        p.set_edgecolor('C1')

    ax.set_xlim(0, 90)
    ax.set_xlabel('inclination (°)')
    ax.set_ylabel('count of event pairs')
    ax.legend(title='side of maximum', frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig(f'{OUT2}hist.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.rcdefaults()

def polar_plots(stuff):

#    hist2 = physt.special.polar_histogram(inc[max1>0], np.deg2rad(azi[max1>0]), transformed=True, radial_bins=10, phi_bins=10)
#    ax = hist2.plot.polar_map(show_values=True)
#    print(len(azi[max1<=0]))
#    hist2 = physt.special.polar_histogram(inc[max1<=0], np.deg2rad(azi[max1<=0]), transformed=True, radial_bins=10, phi_bins=10)
#    ax = hist2.plot.polar_map(show_values=True)

    # Create a polar histogram with different binning

    dist, tmax1, tmax2, max1, max2, (snr1, snr2, snr3, snr4), evids, azi, inc, (coords1, coords2), v, (times1, times2), phase1 = repack(stuff.values())
    coords1 = convert_coords2km(coords1, LATLON0)
    coords2 = convert_coords2km(coords2, LATLON0)
    x1, y1, z1 = [np.array(bla) for bla in zip(*coords1)]
    x2, y2, z2 = [np.array(bla) for bla in zip(*coords1)]
    x = 0.5 * (x1 + x2)
    y = 0.5 * (y1 + y2)
    z = 0.5 * (z1 + z2)

    cond = np.logical_and(snr2 <= NSR, dist > 200)
    dist, tmax1, tmax2, max1, max2, snr1, snr2, snr3, snr4, azi, inc, x, y, z = filters(cond, dist, tmax1, tmax2, max1, max2, snr1, snr2, snr3, snr4, azi, inc, x, y, z)


    nr = 9
    ntheta = 18
    r_edges = np.linspace(0, 90, nr + 1)
    theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
    Theta, R = np.meshgrid(theta_edges, r_edges)

    H1, _, _ = np.histogram2d(inc[max1>0], np.deg2rad(azi[max1>0]), [r_edges, theta_edges])
    H2, _, _ = np.histogram2d(inc[max1<=0], np.deg2rad(azi[max1<=0]), [r_edges, theta_edges])
    H1[0, :] = np.sum(H1[0, :])
    H2[0, :] = np.sum(H2[0, :])
    HH = (H1 - H2) / (H1 + H2)
    HH[H1 + H2 < 6] = np.nan

    bbfname = 'data/focal_mechanism_2018swarm.txt'
    bbs = np.genfromtxt(bbfname, names=True)
    sdrs = [sdr for _, _, _, _, _, *sdr in bbs]
    sdr_mean = np.median(sdrs, axis=0)

    def plot_bb(ax):
        axb = fig.add_axes(ax.get_position().bounds, aspect='equal')
        axb.axison = False
        b = beach(sdr_mean, width=20, nofill=True, linewidth=1)
        axb.add_collection(b)
        axb.set_xlim(-10, 10)
        axb.set_ylim(-10, 10)
        return axb

    fig = plt.figure(figsize=(10, 5))
    kw = dict(edgecolors='face', linewidths=0.01)

    ax2 = fig.add_subplot(132, polar=True)
    cb_kw = dict(shrink=0.25, panchor=(0.95, 0.95), aspect=10, use_gridspec=False)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    vmax = np.nanmax(np.abs(HH))
    cmap = 'coolwarm_r'
    im = ax2.pcolormesh(Theta, R, HH, cmap=cmap, vmax=vmax, vmin=-vmax, **kw)
    thetaline = np.deg2rad([320, 340, 360, 360, 340, 320, 320])
    rline = [20, 20, 20, 50, 50, 50, 20]
    ax2.plot(thetaline, rline, color='0.4')
    ax2.set_yticks([15, 35, 55, 75])
    ax2.set_yticklabels(['20°', '40°', '60°', '80°'])
    plt.colorbar(im, ax=ax2, **cb_kw)
#    ax2b = fig.add_axes(ax2.get_position().bounds)
#    plot_farfield(sdr_mean, typ=None, ax=ax2b, plot_pt=True)
    plot_bb(ax2)
    plot_pnt(sdr_mean, ax2)

    ax3 = fig.add_subplot(133, polar=True, sharex=ax2, sharey=ax2)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    im = ax3.pcolormesh(Theta, R, H1 + H2, cmap='magma_r', zorder=-1, **kw)
    ax3.plot(thetaline, rline, color='0.4')
    ax3.set_yticks([15, 35, 55, 75])
    ax3.set_yticklabels(['20°', '40°', '60°', '80°'])
    plt.colorbar(im, ax=ax3, **cb_kw)
#    plot_farfield(sdr_mean, typ=None, ax=ax3, plot_pt=True)
    plot_bb(ax3)
    plot_pnt(sdr_mean, ax3)

    b1 = ax2.get_position().bounds
    b2 = ax3.get_position().bounds
#    ax1 = fig.add_subplot(131, aspect='equal')
    ax1 = fig.add_axes((2*b1[0]-b2[0],) + b1[1:], aspect='equal')
    ax1b, ax1c = plot_farfield(sdr_mean, typ='S', points=nice_points(), ax=ax1, scale=8, plot_axis='PNT')
    for sdr in sdrs:
        b = beach(sdr, width=20, edgecolor='0.7', nofill=True, linewidth=0.5)
        ax1c.add_collection(b)
#        plot_pnt(sdr, ax1b, 'PNT', color='0.7', label='', zorder=-1)
    plot_bb(ax1c)

    annokw = dict(xy=(0.5, -0.2), xycoords='axes fraction', ha='center')
    annokw2 = dict(xy=(1.02, -0.25), xycoords='axes fraction', ha='right')
    annokw3 = dict(xy=(0, 1), xycoords='axes fraction')
    annokw4 = dict(xy=(-0.3, 1), xycoords='axes fraction')
    ax1.annotate('S wave radiation pattern\nfocal mechanisms', **annokw)
    ax2.annotate('polarity\nof maxima', **annokw2)
    ax3.annotate('number\nof pairs', **annokw2)
    ax1.annotate('a)', **annokw3)
    ax2.annotate('b)', **annokw4)
    ax3.annotate('c)', **annokw4)

    fig.savefig(f'{OUT2}focal.pdf', bbox_inches='tight', pad_inches=0.1)



    H1, _, _ = np.histogram2d(inc[tmax1>0], np.deg2rad(azi[tmax1>0]), [r_edges, theta_edges])
    H2, _, _ = np.histogram2d(inc[tmax1<=0], np.deg2rad(azi[tmax1<=0]), [r_edges, theta_edges])
    H1[0, :] = np.sum(H1[0, :])
    H2[0, :] = np.sum(H2[0, :])
    HH = (H1 - H2) / (H1 + H2)
    ind = H1 + H2 < 10
#    HH[ind] = np.nan

    from scipy.stats import binned_statistic_2d

    plt.figure()
    ax = plt.subplot(331, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    vmax = np.nanmax(np.abs(HH))
    im = plt.pcolormesh(Theta, R, HH, cmap='coolwarm', vmax=vmax, vmin=-vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('direction')

    ax = plt.subplot(332, polar=True, sharex=ax, sharey=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    im = plt.pcolormesh(Theta, R, H1 + H2, cmap='magma_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('number')

    ax = plt.subplot(333, polar=True, sharex=ax, sharey=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    stats, *_ = binned_statistic_2d(inc, np.deg2rad(azi), snr2, statistic='median', bins=[r_edges, theta_edges], expand_binnumbers=False)
    stats[ind] = np.nan
    im = plt.pcolormesh(Theta, R, stats, cmap='magma_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('noise')

    ax = plt.subplot(334, polar=True, sharex=ax, sharey=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    stats, *_ = binned_statistic_2d(inc, np.deg2rad(azi), x, statistic='median', bins=[r_edges, theta_edges], expand_binnumbers=False)
    stats[ind] = np.nan
    im = plt.pcolormesh(Theta, R, stats, cmap='magma_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x')

    ax = plt.subplot(335, polar=True, sharex=ax, sharey=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    stats, *_ = binned_statistic_2d(inc, np.deg2rad(azi), y, statistic='median', bins=[r_edges, theta_edges], expand_binnumbers=False)
    stats[ind] = np.nan
    im = plt.pcolormesh(Theta, R, stats, cmap='magma_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('y')

    ax = plt.subplot(336, polar=True, sharex=ax, sharey=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    stats, *_ = binned_statistic_2d(inc, np.deg2rad(azi), z, statistic='median', bins=[r_edges, theta_edges], expand_binnumbers=False)
    stats[ind] = np.nan
    im = plt.pcolormesh(Theta, R, stats, cmap='magma_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('z')

    ax = plt.subplot(337, polar=True, sharex=ax, sharey=ax)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    stats, *_ = binned_statistic_2d(inc, np.deg2rad(azi), dist/1000/np.abs(tmax1), statistic='median', bins=[r_edges, theta_edges], expand_binnumbers=False)
    stats[ind] = np.nan
    im = plt.pcolormesh(Theta, R, stats, cmap='magma_r', vmax=5)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('velocity')
    plt.savefig(f'{OUT}focal_plots.pdf', bbox_inches='tight', pad_inches=0.1)


#NAME = 'tmp/ccs_stack_2018_mag>1.8_q2_246events_hp10Hz_1bit_dist<1km_l.h5'
#NAME = 'tmp/ccs_stack_2018_mag>1.8_q3_hp10Hz_1bit_dist<1km_l.h5'
#NAME = 'tmp/ccs_stack_2018_mag>1.8_q3_hp20Hz_1bit_dist<1km_l.h5'
#NAME = 'tmp/ccs_stack_2018_mag>1.8_q3_10Hz-40Hz_1bit_dist<1km_pw2.h5'
#NAME = 'tmp/ccs_stack_2018_mag>1.8_q3_10Hz-40Hz_Pcoda_envelope_dist<1km_pw2.h5'
#NAME = 'tmp/ccs_stack_2018_mag>1.8_q3_10Hz-40Hz_P_None_dist<1km_pw2.h5'
#NAME = 'tmp/ccs_2018_mag>1.8_q3_10Hz-40Hz_P_None_dist<1km_pw2.h5'
NAME = 'tmp/ccs_stack_2018_mag>1.8_q3_10Hz-40Hz_Scoda_envelope_dist<1km_pw2.h5'
NAMED = 'tmp/ccs_stack_2018_mag>1.8_q3_10Hz-40Hz_S_None_dist<1km_pw2.h5'
name = NAME.split('ccs_stack_2018_')[1].split('.h5')[0]
OUT = 'tmp/'
OUT2 = 'figs/'
PKL_FILE1 = 'tmp/stuff.pkl'
PKL_FILE2 = 'tmp/stuff_onlymax.pkl'
V = 3.6
#V = 8
DIST = 1

NOISE_IND = 7
NSR = 0.1

if __name__ == '__main__':

    plot_corr_vs_dist2times()
    ccs_stack = read(NAME, 'H5')
    print(f'len stream {len(ccs_stack)}')

    add_distbin(ccs_stack, 1, 21)

    ccs_stack_dist = copy(ccs_stack).stack('{distbin}').sort(['distbin'])
    for tr in ccs_stack_dist:
        tr.stats.dist = tr.stats.distbin

#    ccs_stack_dist_norm = ccs_stack.copy().normalize().stack('{distbin}').sort(['distbin'])
#    for tr in ccs_stack_dist_norm:
#        tr.stats.dist = tr.stats.distbin
#
#    plot_corr_vs_dist_hist(ccs_stack_dist_norm, figsize=(10, 8), vmax=0.3, v=V, xlim=0.8*DIST)
#    plt.ylabel('event distance (m)')
#    plt.savefig(f'{OUT}corr_vs_dist_hist_norm_{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
#    plot_corr_vs_dist(ccs_stack_dist.copy().normalize(), figsize=(10, 8), annotate=False, v=V, xlim=0.8*DIST)
#    plt.ylabel('event distance (m)')
#    plt.savefig(f'{OUT}corr_vs_dist_wigg_{name}.png', bbox_inches='tight', pad_inches=0.1)
#    plot_corr_vs_dist(ccs_stack_dist_norm.copy().normalize(), figsize=(10, 8), annotate=False, v=V, xlim=0.8*DIST)
#    plt.ylabel('event distance (m)')
#    plt.savefig(f'{OUT}corr_vs_dist_wigg_norm_{name}.png', dpi=300)

#    stuff = get_stuff(trim2(ccs_stack.copy(), -0.5, 0.5, 'mid'), trim2(ccs_stack.copy(), -0.5, 0.5, 'mid'))
#    with open(PKL_FILE1, 'wb') as f:
#        pickle.dump(stuff, f, protocol=pickle.HIGHEST_PROTOCOL)
#    stuff2 = get_stuff(trim2(ccs_stack.copy(), -0.5, 0.5, 'mid'), trim2(ccs_stack.copy(), -0.5, 0.5, 'mid'), only_maxima=True)
#    with open(PKL_FILE2, 'wb') as f:
#        pickle.dump(stuff2, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PKL_FILE1, 'rb') as f:
        stuff = pickle.load(f)
    with open(PKL_FILE2, 'rb') as f:
        stuff2 = pickle.load(f)
    print(f'len stuff {len(stuff)}')

    plot_maxima_vs_dist(stuff, figsize=(10, 8), v=V)
    plt.savefig(f'{OUT}maxima_vs_dist2_{name}.png', dpi=300)
    plt.savefig(f'{OUT2}maxima_vs_dist.pdf')

    plot_maxima_vs_dist(stuff2, figsize=(10, 8), v=V)
    plt.savefig(f'{OUT}maxima_vs_dist2_{name}_only_max.png', dpi=300)
#
    polar_plots(stuff)
    plot_hists(stuff, figsize=(19, 10.5))
    plt.show()


#    make_crazy_plot(stuff, figsize=(16, 8))
#    plt.savefig(f'{OUT}crazy_plot_{name}.png', dpi=300)

#    for key, stream in ccs_stack._groupby('{distbin}').items():
#        if make_crazy_plot(stream, figsize=(16, 8)):
#            plt.savefig(f'{OUT}crazy_plot2_{key}_{name}.png', dpi=300)

#    plot_stuff_map_depth(stuff, plot_lines=True)

#    c1, c2 = analyze_stuff(stuff)
#    for evid in c1.keys():
#        ccs_stack_some = [tr for tr in ccs_stack if evid in tr.stats.stack.group]
#        plot_maxima_vs_dist_event(ccs_stack_some, evid, v=4000)
#        plt.savefig(f'{OUT}/zzz/{name}_maxima_vs_dist_{evid}.png', dpi=300)
#        plt.close()
#
#    dists = [(int(float(tr.stats.stack.group)), tr.stats.stack.count) for tr in ccs_stack_dist]
#    print(dists)
#    for d in sorted(dists):
#        traces = [tr for tr in ccs_stack if tr.stats.distbin == d[0]]
#        traces = sorted(traces, key=lambda tr: tr.stats.dist)
#        ccssub = ccs_stack.__class__(traces)
#        plot_corr(ccssub, figsize=(20, 20))
#        plt.savefig(f'{OUT}/yyy/{name}_{d[0]:03d}.png')
#        plt.close()
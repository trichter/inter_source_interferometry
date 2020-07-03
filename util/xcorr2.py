# Copyright 2020 Tom Eulenfeld, MIT license

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.cross_correlation import correlate
import utm


def correlate_phase(data1, data2, shift, demean=True, normalize=True):
    from scipy.signal import hilbert
    from scipy.fftpack import next_fast_len
    assert len(data1) == len(data2)
    nfft = next_fast_len(len(data1))
    if demean:
        data1 = data1 - np.mean(data1)
        data2 = data2 - np.mean(data2)
    sig1 = hilbert(data1, N=nfft)[:len(data1)]
    sig2 = hilbert(data2, N=nfft)[:len(data2)]
    phi1 = np.angle(sig1)
    phi2 = np.angle(sig2)
    def phase_stack(phi1, phi2, shift):
        s1 = max(0, shift)
        s2 = min(0, shift)
        N = len(phi1)
        assert len(phi2) == N
        return np.sum(np.abs(np.cos((phi1[s1:N+s2] - phi2[-s2:N-s1]) / 2)) -
                      np.abs(np.sin((phi1[s1:N+s2] - phi2[-s2:N-s1]) / 2)))

    cc = [phase_stack(phi1, phi2, s) for s in range(-shift, shift + 1)]
    cc = np.array(cc)
    if normalize:
        cc = cc / len(data1)
    return cc


def coord2m(lat, lon, dep):
    x, y = utm.from_latlon(lat, lon)[:2]
    return x, y, dep


def calc_angle(a, b, c):
    a = np.array(coord2m(*a))
    b = np.array(coord2m(*b))
    c = np.array(coord2m(*c))
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def correlate_traces(tr1, tr2, shift, abuse_seedid=False,
                     use_headers=set(), use_seedid_headers=True,
                     calc_header=None, phase=False,
                     **kwargs):
    """
    Return trace of cross-correlation of two input traces

    :param tr1,tr2: two |Trace| objects
    :param shift: maximal shift in correlation in seconds
    """
    if use_seedid_headers:
        seedid_headers = {'network', 'station', 'location', 'channel'}
        use_headers = set(use_headers) | seedid_headers
    sr = tr1.stats.sampling_rate
    assert sr == tr2.stats.sampling_rate
    header = {k: v for k, v in tr1.stats.items() if tr2.stats.get(k) == v and k != 'npts'}
    for k in use_headers:
        if k in tr1.stats:
            header[k + '1'] = tr1.stats[k]
        if k in tr2.stats:
            header[k + '2'] = tr2.stats[k]
    if 'channel' not in header:
        c1 = tr1.stats.channel
        c2 = tr2.stats.channel
        if c1 != '' and c2 != '' and c1[:-1] == c2[:-1]:
            header['channel'] = c1[:-1] + '?'
            header['channel'] = c1[:-1] + c1[-1] + c2[-1]
    st1 = tr1.stats.starttime
    st2 = tr2.stats.starttime
    len1 = tr1.stats.endtime - st1
    len2 = tr2.stats.endtime - st2
    if (st1 + len1 / 2) - (st2 + len2 / 2) < 0.1:
        header['starttime'] = st1 + len1 / 2
    corrf = correlate_phase if phase else correlate
    xdata = corrf(tr1.data, tr2.data, int(round(shift * sr)), **kwargs)
    tr = Trace(data=xdata, header=header)
    if abuse_seedid:
        n1, s1, l1, c1 = tr1.id.split('.')
        n2, s2, l2, c2 = tr2.id.split('.')
        tr.id = '.'.join(s1, c1, s2, c2)
    if calc_header == 'event' and 'elon' in tr1.stats:
        s1 = tr1.stats
        s2 = tr2.stats
        args = (s1.elat, s1.elon, s2.elat, s2.elon)
        dist, azi, baz = gps2dist_azimuth(*args)
        dpdif = (s2.edep - s1.edep) * 1000
        tr.stats.dist = (dist ** 2 + dpdif**2) ** 0.5  # dist in meter
        tr.stats.azi = azi
        tr.stats.baz = baz
        tr.stats.inc = np.rad2deg(np.arctan2(dist, dpdif))
        # only valid if event 1 is above event 2
        assert tr.stats.inc <= 90.
        if 'slat' in s1:
            a = s1.elat, s1.elon, s1.edep * 1000
            b = s2.elat, s2.elon, s2.edep * 1000
            c = s1.slat, s1.slon, 0
            tr.stats.angle12s = calc_angle(a, b, c)
            tr.stats.angle21s = calc_angle(b, a, c)
    elif calc_header == 'station' and 'slon' in tr1.stats:
        args = (tr1.stats.slat, tr1.stats.slon, tr2.stats.slat, tr2.stats.slon)
        dist, azi, baz = gps2dist_azimuth(*args)
        tr.stats.dist = dist / 1000  # dist in km
        tr.stats.azi = azi
        tr.stats.baz = baz
    return tr


def correlate_streams(stream1, stream2, shift, **kwargs):
    traces = [correlate_traces(tr1, tr2, shift, **kwargs)
              for tr1, tr2 in zip(stream1, stream2)]
    return Stream(traces)


def keypress(event, fig, ax, alines):
    if event.inaxes != ax:
        return
    print('You pressed ' + event.key)
    if event.key == 'k':
        print(__doc__)
    elif event.key in 'gb':
        factor = 1.2 if event.key == 'g' else 1/1.2
        for lines, dist in alines:
            lines.set_ydata((lines.get_ydata() - dist) * factor + dist)
    fig.canvas.draw()


def velocity_line(v, ax, times=None, dists=None, lw=1, zorder=None, color='C1', alpha=0.8):
    if times is None:
        times = ax.get_xlim()
    if dists is None:
        dists = ax.get_ylim()
    times = np.array([max(min(times), -max(dists) / v / 1000), 0, min(max(times), max(dists) / v / 1000)])
    ax.plot(times, np.abs(v * 1000 * times), color=color, lw=lw, alpha=alpha, label='%s km/s' % v, zorder=zorder)


def plot_corr_vs_dist(stream, figsize=None, v=3.5, annotate=True, expr='{evid1}-{evid2}', xlim=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    max_ = max(stream.max())
    max_dist = max(tr.stats.dist for tr in stream)
    all_lines = []
    for i, tr in enumerate(stream):
        starttime = tr.stats.starttime
        mid = starttime + (tr.stats.endtime - starttime) / 2
        t = tr.times(reftime=mid)
        scaled_data = tr.stats.dist + tr.data * max_dist / max_ / 50
        (lines,) =ax.plot(t, scaled_data, 'k', lw=1)
        all_lines.append((lines, tr.stats.dist))
        if annotate:
            label = expr.format(**tr.stats)
            ax.annotate(label, (t[-1], tr.stats.dist), (-5, 0),
                        'data', 'offset points', ha='right')
    velocity_line(v, ax, t, [max_dist], lw=2)
    ax.legend()
    ax.set_ylabel('distance (m)')
    ax.set_xlabel('lag time (s)')
    k = lambda x: keypress(x, fig, ax, all_lines)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect('key_press_event', k)
    if xlim:
        plt.xlim(-xlim, xlim)
    plt.ylim(0, None)


def _align_values_for_pcolormesh(x):
    x = list(x)
    dx = np.median(np.diff(x))
    x.append(x[-1] + dx)
    x = [xx - 0.5 * dx for xx in x]
    return x


def plot_corr_vs_dist_hist(stack, vmax=None, cmap='RdBu_r', figsize=None, v=3.5, xlim=None):
    lag_times = stack[0].times()
    lag_times -= lag_times[-1] / 2
    lag_times = _align_values_for_pcolormesh(lag_times)
    data = np.array([tr.data for tr in stack])
    dists = [tr.stats.distbin for tr in stack]
    if vmax is None:
        vmax = 0.8 * np.max(np.abs(data))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.1, 0.75, 0.75])
    velocity_line(v, ax, lag_times, [max(dists)])
    ax.legend()
    cax = fig.add_axes([0.91, 0.375, 0.008, 0.25])
    mesh = ax.pcolormesh(lag_times, dists, data, cmap=cmap,
                         vmin=-vmax, vmax=vmax)
    fig.colorbar(mesh, cax)
    ax.set_ylabel('distance (m)')
    ax.set_xlabel('time (s)')
    plt.sca(ax)
    if xlim:
        plt.xlim(-xlim, xlim)
    plt.ylim(0, None)


# https://stackoverflow.com/a/31364297
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
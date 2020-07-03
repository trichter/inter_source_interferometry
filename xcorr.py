# Copyright 2020 Tom Eulenfeld, MIT license

import collections
from copy import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy import read, read_events
from obspy.geodetics import gps2dist_azimuth
import obspyh5
from tqdm import tqdm

from load_data import iter_data
from util.events import event2list
from util.signal import trim2
from util.xcorr2 import correlate_traces


def plot_stream(stream):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_ = max(stream.max())
    for i, tr in enumerate(stream):
        times = tr.times(reftime=tr.stats.event_time)
        ax.plot(times, i + tr.data/max_*2)


def print_(stream, s=False):
    print(stream)
    m = [(tr.stats.dist, tr.stats.angle12s, tr.stats.angle21s) for tr in stream]
    if s:
        m = sorted(m)
    for ml in m:
        print(ml)


def _get_dist(lat1, lon1, dep1, lat2, lon2, dep2):
    dist, azi, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dpdif = (dep1 - dep2) * 1000
    return (dist ** 2 + dpdif**2) ** 0.5  # dist in meter


def correlate_stream(stream, tw='Scoda', timenorm=None, max_dist=None, tw_len=None, max_lag=2, filter=None,
                     max_dist_diff=None, **kw):
    if FILTER:
        if FILTER['freqmax'] > 100:
            stream.filter('highpass', freq=FILTER['freqmin'])
        else:
            stream.filter(**FILTER)
    for tr in stream:
        if timenorm == '1bit':
            tr.data = np.sign(tr.data)
        elif timenorm == 'envelope':
            from stools.signal import envelope
            tr.data = tr.data / envelope(tr.data)
    streams = collections.defaultdict(obspy.Stream)
    for tr in stream:
        streams[tr.stats.evid].append(tr)
    i = 0
    j = 0
    traces = []
    for stream1, stream2 in tqdm(list(itertools.combinations(streams.values(), 2))):
        if EVID_PAIRS is not None and '-'.join([stream1[0].stats.evid, stream2[0].stats.evid]) not in EVID_PAIRS:
            continue
        # event 1 above event 2
        if stream1[0].stats.edep > stream2[0].stats.edep:
            stream1, stream2 = stream2, stream1
        for tr1, tr2 in itertools.product(stream1, stream2):
            if tr1.id == tr2.id:
#            if tr1.stats.station == tr2.stats.station:
                if max_dist is not None:
                    s1 = tr1.stats
                    s2 = tr2.stats
                    dist = _get_dist(s1.elat, s1.elon, s1.edep, s2.elat, s2.elon, s2.edep)
                    if dist > max_dist * 1000:
                        j += 1
#                        print('distance', s1.evid, s2.evid, dist2)
                        continue
                if max_dist_diff is not None:
                    s1 = tr1.stats
                    s2 = tr2.stats
                    dist = _get_dist(s1.elat, s1.elon, s1.edep, s2.elat, s2.elon, s2.edep)  # between events
                    dist1 = _get_dist(s1.elat, s1.elon, s1.edep, s1.slat, s1.slon, 0)  # betweeen event 1 and station1
                    dist2 = _get_dist(s2.elat, s2.elon, s2.edep, s1.slat, s1.slon, 0)  # betweeen event 2 and station1
                    dpdif = abs(s1.edep -s2.edep)
                    if dpdif == 0:
                        continue
                    dist01 = dist / dpdif * s1.edep  # between event 1 and surface
                    dist02 = dist / dpdif * s2.edep  # between event 2 and surface
#                    print(abs(dist01 - dist1), abs(dist02 - dist2))
                    if abs(dist01 - dist1) > max_dist_diff or abs(dist02 - dist2) > max_dist_diff:
                        j += 1
                        continue
                if tw == 'Scoda':
                    t1 = max(tr1.stats.twScoda[0], tr2.stats.twScoda[0])
                    t2 = min(tr1.stats.twScoda[1], tr2.stats.twScoda[1])
                    if tw_len is not None and t2 - t1 < tw_len:
                        j += 1
    #                    print('tw', tr1.id, s1.evid, s2.evid)
                        continue
                    et1 = tr1.stats.event_time
                    tr1 = tr1.copy().trim(et1 + t1, et1 + t2, pad=True, fill_value=0.0)
                    et2 = tr2.stats.event_time
                    tr2 = tr2.copy().trim(et2 + t1, et2 + t2, pad=True, fill_value=0.0)
#                    if (tr1.stats.endtime - tr1.stats.starttime < 0.9 * tw_len or
#                            tr2.stats.endtime - tr2.stats.starttime < 0.9 * tw_len):
#                        j += 1
#    #                    print('tw 2', s1.evid, s2.evid)
#                        continue
                else:
                    tw1 = tr1.stats.get('tw' + tw)
                    tw2 = tr2.stats.get('tw' + tw)
                    t1 = min(tw1[0], tw2[0])
                    t2 = max(tw1[1], tw2[1])
                    if t2 - t1 < 0.5:
                        j += 1
                        continue
                    et1 = tr1.stats.event_time
                    tr1 = tr1.copy().trim(et1 + t1, et1 + t2, pad=True, fill_value=0.0)
                    et2 = tr2.stats.event_time
                    tr2 = tr2.copy().trim(et2 + t1, et2 + t2, pad=True, fill_value=0.0)
                if np.all(tr1.data == 0) or np.all(tr2.data == 0):
                    j += 1
                    continue
                if len(tr1) > len(tr2):
                    tr1.data = tr1.data[:len(tr2)]
                elif len(tr2) > len(tr1):
                    tr2.data = tr2.data[:len(tr1)]
                hack = False
                if hack:
                    tanf = t1
                    while tanf < t2:
                        trx = correlate_traces(tr1.slice(et1+tanf, et1+tanf + 5), tr2.slice(et2+tanf, et2+tanf +5), max_lag, calc_header='event', use_headers=('evid', 'elat', 'elon', 'edep', 'mag', 'event_time'), **kw)
                        trx.stats.tw = (tanf, tanf + 10)
                        trx.stats.tw_len = 10
                        traces.append(trx)
                        tanf += 10
                else:
                    tr = correlate_traces(tr1, tr2, max_lag, calc_header='event', use_headers=('evid', 'elat', 'elon', 'edep', 'mag', 'event_time'), **kw)
                    tr.stats.tw = (t1, t2)
                    tr.stats.tw_len = t2 - t1
                    if max_dist_diff is not None:
                        tr.stats.dist1 = dist1
                        tr.stats.dist2 = dist2
                        tr.stats.dist01 = dist01
                        tr.stats.dist02 = dist02
                        tr.stats.distx = max(abs(dist1 - dist01), abs(dist2 - dist02))
                    traces.append(tr)
                i += 1
        if len(traces) > 0:
            yield obspy.Stream(traces)
        traces = []
    if i + j >0:
        print(f'correlations: {i} succesful, {j} discarded, {100*i/(i+j):.2f}%')
    else:
        print('ups')


def correlate_stream2(*args, **kwargs):
    ccs = obspy.Stream()
    for trcs in correlate_stream(*args, **kwargs):
        ccs.traces.extend(trcs)
    return ccs


def load_data_add_meta(data, events, coords, alldata=False):
    stream = obspy.Stream()
    lens = []
    for s, e in tqdm(iter_data(events, alldata=alldata), total=len(events)):
         id_, otime, lon, lat, dep, mag, picks = event2list(e)
         for tr in s:
             tr.stats.event_time = otime
             tr.stats.evid = id_
             tr.stats.elon = lon
             tr.stats.elat = lat
             tr.stats.edep = dep
             tr.stats.mag = mag
             tr.stats.selev = 0
             try:
                 tr.stats.slat, tr.stats.slon = coords[tr.stats.station]
             except Exception as ex:
                 print(ex, tr.id)
                 continue
         l1 = len(stream)
         stream += s
         l2 = len(stream)
         assert l2 - l1 == len(s)
         lens.append(len(s))
         if len(s) < 8*3:
             from IPython import embed
             embed()
    from collections import Counter
    print('Counter(lens)', Counter(lens))
    print('len(stream)', len(stream))
    return stream


def picks2stream(stream, events):
    from load_data import get_picks
    picks, relpicks, _, _ = get_picks(events)
    for tr in stream:
        sta = tr.stats.station
        id_ = tr.stats.evid
        tr.stats.spick = relpicks[id_][(sta, 'S')][0]
        tr.stats.ppick = relpicks[id_][(sta, 'P')][0]


def tw2stream(stream):
    from load_data import tw_from_qc_file, stacomp
    tw = tw_from_qc_file()
    for tr in stream:
        id_ = tr.stats.evid
        pkey = stacomp(tr.id)
        tr.stats.tw = tw[id_][2][pkey][:2]
        tr.stats.quality = tw[id_][0]
        tr.stats.quality_str = tw[id_][1]
        tr.stats.tw_reason = tw[id_][2][pkey][2]
        tr.stats.noise = tw[id_][2][pkey][3]

        tr.stats.twP = (tr.stats.ppick, min(tr.stats.spick, tr.stats.ppick + 0.5))
#        tr.stats.twPcoda = (min(tr.stats.spick, tr.stats.ppick + 0.5), tr.stats.spick)
        tr.stats.twS = (tr.stats.spick, tr.stats.spick + 2)
        tr.stats.twScoda = tr.stats.tw
        tr.stats.twP = (tr.stats.ppick, tr.stats.spick - 0.02)


def filter_events(events, quality):
    from load_data import tw_from_qc_file, select_events
    tw = tw_from_qc_file()
    return select_events(events, tw, quality)


def create_stream(events, coords, write=True, name_data_out=None, alldata=False):
    stream = load_data_add_meta(DATAFILES, events, coords, alldata=alldata)
    for tr in stream:
        if tr.stats.sampling_rate != 250:
            tr.interpolate(250)
    trim2(stream, *TRIM1, 'event_time', check_npts=False)
    picks2stream(stream, events)
    tw2stream(stream)
    if write:
        stream.write(name_data_out + '.h5', 'H5')
    return stream


def load_stream():
    name_data_in = OUT + f'{YEAR}_mag>{MAG}_q{QU}_*'
    return read(name_data_in + '.h5')


def add_distbin(stream, max_dist, n=51):
    dists = np.linspace(0, 1000 * max_dist, n)
    dists_mean = (dists[:-1] + dists[1:]) / 2
    for tr in stream:
        ind = np.digitize(tr.stats.dist, dists)
        tr.stats.distbin = dists_mean[ind-1]


def _select_max(stream):
    traces = []
    def _bla(tr):
        return abs(np.argmax(np.abs(tr.data)) - len(tr.data) / 2 + 0.5)

    for k, st in tqdm(stream._groupby('{evid1}-{evid2}').items()):
        st = st.copy()
        trcs = st
        v = 0.7 * np.max(np.abs(trcs.max()))
        trcs = [tr for tr in trcs if np.max(np.abs(tr.data)) > v]
        bb = 0.9 * max(_bla(tr) for tr in trcs)
        trcs = [tr for tr in trcs if _bla(tr) >= bb]
        st.traces = trcs
        st.stack('{evid1}-{evid2}')
        assert len(st) == 1
        traces.append(st[0])
    stream.traces = traces
    return stream


def run_xcorr(stream=None, tw='Scoda', timenorm=None, write=False, write_all=False, **kw):
    if stream is None:
        stream = load_stream()
    name = f"{YEAR}_mag>{MAG}_q{QU}_{FILTER['freqmin']:02.0f}Hz-{FILTER['freqmax']:02.0f}Hz_{tw}_{timenorm}_dist<{DIST}km_{STACK[0]}{STACK[1]}"
    ccs = correlate_stream2(stream, tw=tw, timenorm=timenorm, max_dist=DIST, tw_len=TW_LEN, **kw)
    add_distbin(ccs, DIST)
    if STACK != 'max':
        if 'coda' in tw:
            ccs_stack1 = tw_stack(ccs.copy(), '{evid1}-{evid2}', stack_type=STACK)
        else:
            ccs_stack1 = copy(ccs).stack('{evid1}-{evid2}', stack_type=STACK)
    else:
        ccs_stack1 = _select_max(copy(ccs))
    if write:
        obspyh5.set_index('waveforms/{stack.group}')
        ccs_stack1.write(f'{OUT}ccs_stack_{name}.h5', 'H5')
    if write_all:
        obspyh5.set_index('waveforms/{evid1}-{evid2}/{network}.{station}.{location}.{channel}')
        ccs.write(f'{OUT}ccs_{name}.h5', 'H5', ignore=('processing', ))
    return ccs, ccs_stack1


def plot_corr(stream, annotate=True, expr='{station}.{channel} {evid1}-{evid2}',
              expr2='{evid1}-{evid2}  dist: {dist:.0f}m  azi:{azi:.0f}°  inc:{inc:.0f}°',
              figsize=None, v=3.6, ax=None,
              size1='small', size2='medium'):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    max_ = np.max(np.abs(stream.max()))
    stream = stream.copy()
    trim2(stream, -0.5, 0.5, 'mid')
#    stream.traces.append(copy(stream).stack(stack_type=STACK)[0])
    stream.traces.append(tw_stack(stream.copy(), stack_type=STACK)[0])
    from matplotlib.patches import Rectangle
    rect = Rectangle((0.36, -1), 0.15, len(stream)+1, fc='white', alpha=0.5, ec='none', zorder=10)
    ax.add_patch(rect)
    for i, tr in enumerate(stream):
        starttime = tr.stats.starttime
        mid = starttime + (tr.stats.endtime - starttime) / 2
        t = tr.times(reftime=mid)
        is_stack = i == len(stream) - 1
        plot_kw = dict(color='k', lw=1)
        if is_stack:
            max_ = np.max(np.abs(tr.data))
            plot_kw = dict(color='C0', alpha=0.8, lw=2)
        ax.plot(t, i + tr.data/max_*1.5, **plot_kw)
        if annotate:
            try:
                if is_stack:
                    label = 'stack'
                else:
                    label = expr.format(**tr.stats)
            except KeyError:
                pass
            else:
                ax.annotate(label, (t[-1], i), (-5, 0),
                            'data', 'offset points',
                            ha='right', size=size1, zorder=12)
    dist = stream[0].stats.dist
    tt = dist / v / 1000
    ax.axvline(0, color='0.3', alpha=0.5)
    ax.axvline(-tt, color='C1', alpha=0.8, label='v=%.1fkm/s' % v)
    ax.axvline(tt, color='C1', alpha=0.8)
    ax.legend(loc='lower left', fontsize='medium')
    ax.set_xlabel('lag time (s)')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-1, None)
    ax.set_yticks([])
    if expr2 is not None:
        ax.annotate(expr2.format(**stream[0].stats), (t[0], len(stream)), (5, 10),
                    'data', 'offset points', size=size2)

def plot_xcorrs(stream, tw='Scoda', timenorm=None, **kw):
    v = 3.6 if 'S' in tw else 6
    name = f"{FILTER['freqmin']:02.0f}Hz-{FILTER['freqmax']:02.0f}Hz_{tw}_{timenorm}_{STACK[0]}{STACK[1]}"
    for ccssub in correlate_stream(stream, tw=tw, timenorm=timenorm, max_dist=DIST, tw_len=TW_LEN, **kw):
        ccssub.sort(['angle12s'])
        corrid = ccssub[0].stats.evid1 + '-' + ccssub[0].stats.evid2
#        plot_corr(ccssub, expr='{station}.{channel} <12s={angle12s:3.0f} tw:{tw_len:.0f}s', figsize=(20, 20), v=v)
        plot_corr(ccssub, expr='{station}.{channel}', figsize=(10, 10), v=v)
        plt.savefig(f'{OUT}/{name}_{corrid}.png')
        plt.close()


def plot_xcorrs_pub(stream, tw='Scoda', timenorm=None, **kw):
    plt.rc('font', size=12)
    streams = list(correlate_stream(stream, tw=tw, timenorm=timenorm, max_dist=DIST, tw_len=TW_LEN, **kw))
    assert len(streams) == 2
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1)
    plot_corr(streams[1].sort(), expr='{station}.{channel}', ax=ax1, v=3.6, expr2='a)  event pair: {evid1}-{evid2}  distance: {dist:.0f}m')
    plot_corr(streams[0].sort(), expr='{station}.{channel}', ax=ax2, v=3.6, expr2='b)  event pair: {evid1}-{evid2}  distance: {dist:.0f}m')
    plt.tight_layout(pad=2)
    fig.savefig(f'{OUT2}/corr.pdf')
    plt.rcdefaults()
#    plt.close()


def tw_stack(stream, group_by='all', **kw):
    tws = {}
    for k, s in stream._groupby(group_by).items():
        tws[k] = sum([tr.stats.tw[1] - tr.stats.tw[0] for tr in s])
        assert tws[k] > 0
        for tr in s:
            tr.data = tr.data * (tr.stats.tw[1] - tr.stats.tw[0]) * len(s)
    stream.stack(group_by, **kw)
    for tr in stream:
        tr.data = tr.data / tws[tr.stats.stack.group]
    return stream


MAG = 1.8
FILTER = dict(type='bandpass', freqmin=10, freqmax=40)
TRIM1 = (-20, 80)
DIST = 1
TW_LEN = 10

DATAFILES = 'data/waveforms/{id}_{sta}_?H?.mseed'
DATA = '/home/eule/data/datasets/webnet/'
DATAFILES = DATA + 'data*/{id}_{sta}_?H?.mseed'

YEAR = 2018
#STACK = 'linear'
#STACK = 'max'
STACK = ('pw', 2)
STA = '*'
OUT = 'tmp/'
OUT2 = 'figs/'

QU = 3
#SELECT_IDS = ['201855607', '201875354', '201846343', '201874528', '201812057', '201813536', '201816074']
SELECT_IDS = None
# negative pair
EVID_PAIRS = ['201812237-201813506', '201813506-201866387', '201815372-201817046',
       '201815372-201822608', '201816096-201821444', '201816624-201822004',
       '201816642-201866387', '201819712-201845973', '201821178-201823346',
       '201821594-201823346', '201824030-201827856', '201827856-201845629',
       '201830842-201837228', '20183359-20186614', '201834508-201842988',
       '201834666-201845973', '201835050-201845605', '201835220-201838048',
       '201835220-201838074', '201835220-201840856', '201836160-201838074',
       '201843006-201845973', '201845973-201846013', '201845973-201851822',
       '201846343-201866387', '201847953-201812057', '201855859-201863130',
       '201856935-201865574', '201858578-201856087', '201858578-201865574',
       '201859387-201862742', '201861630-201862742', '20189682-201846343']
# positive pair
EVID_PAIRS = ['201811100-201832446', '201812057-201816074', '201812057-201839137',
       '201812497-201813518', '201812497-201816642', '201813506-201832446',
       '201813518-201839137', '201814705-201815372', '201815314-201817046',
       '201816074-201816096', '201816074-201816642', '201816074-201821600',
       '201816624-201834508', '201816642-201839137', '201816718-201866387',
       '201817042-201819712', '201818492-201822608', '201821178-201821626',
       '201821266-201821408', '201821266-201823556', '201821458-201823556',
       '201821600-201821626', '201821600-201822004', '201822004-201834530',
       '201822608-201834508', '201823988-201827856', '201823988-201837010',
       '201823988-201845605', '201827856-201830842', '201827856-201837218',
       '201830842-201834588', '201830842-201835050', '201830842-201835220',
       '20183359-20185009', '201834508-201835040', '201834508-201835050',
       '201834508-201838074', '201834508-201841526', '201834530-201837650',
       '201834666-201835748', '201834666-201846013', '201835040-201838048',
       '201835040-201838074', '201835050-201835436', '201835220-201845605',
       '201835436-201845605', '201835748-201836160', '201835748-201843006',
       '201837010-201830842', '201837010-201838074', '201837228-201835050',
       '201846343-201872635', '201847435-201847691', '201847435-201851459',
       '20184759-201870893', '201847691-201851459', '201847953-201816718',
       '201851459-201851461', '201855607-201859387', '201855607-201864722',
       '201855675-201856087', '201857277-201859387', '201857277-201864722',
       '201862742-201863130', '20186460-201870893', '201864722-201875354',
       '20186614-20186746', '20186784-201818372', '20186784-20189682',
       '20187260-20189036', '20189682-201870893', '20189946-201812093',
       '20189946-201812237', '20189946-201813506']
# peaks not fitting to velocity
EVID_PAIRS = ['201810202-201812057', '201810202-201812237', '201810202-201812497', '201810202-201813518', '201810202-201813883', '201810202-201816074', '201810202-201821444', '201811100-201812259', '201811100-201812497', '201811100-201813518', '201811100-201813821', '201811100-201821600', '201811100-201839137', '201812057-201818254', '201812057-201821178', '201812057-201821626', '201812057-201822004', '201812057-201837650', '201812057-201866311', '201812093-201813536', '201812093-201813821', '201812093-201814705', '201812093-201815372', '201812093-201821574', '201812093-201834530', '201812237-201812605', '201812237-201813536', '201812237-201836704', '201812237-201871472', '201812259-201813883', '201812259-201818492', '201812259-201820961', '201812259-201822004', '201812497-201812605', '201812497-201821738', '201812497-201822004', '201812605-201813039', '201812605-201815372', '201812605-201816624', '201812605-201872635', '201812605-201874528', '201813039-201813506', '201813039-201818254', '201813039-201820961', '201813039-201821408', '201813039-201821458', '201813039-201821600', '201813039-201821738', '201813039-201822004', '201813039-201822608', '201813039-201823988', '201813039-201829236', '201813039-201832446', '201813039-201834530', '201813039-201837650', '201813039-201871472', '201813506-201813536', '201813506-201813821', '201813506-201815314', '201813506-201815372', '201813506-201821574', '201813506-201834530', '201813506-201872635', '201813518-201813821', '201813518-201823346', '201813518-201871472', '201813536-201821408', '201813536-201821594', '201813536-201827252', '201813536-201832446', '201813536-201871472', '201813536-201877386', '201813821-201816718', '201813821-201821444', '201813821-201832446', '201813821-201866311', '201813883-201818346', '201813883-201834588', '201813883-201871472', '201814705-201821408', '201814705-201821458', '201814705-201821520', '201814705-201824030', '201814705-201827856', '201814705-201835050', '201814705-201840856', '201814705-201866387', '201815314-201819712', '201815314-201821444', '201815314-201821600', '201815314-201834508', '201815314-201840856', '201815314-201872635', '201815372-201823988', '201815372-201832446', '201815372-201840856', '201815372-201871472', '201816074-201818346', '201816074-201834530', '201816074-201871472', '201816096-201816718', '201816096-201820961', '201816096-201821178', '201816096-201821520', '201816096-201827252', '201816096-201837650', '201816624-201816718', '201816624-201821178', '201816624-201821444', '201816624-201827856', '201816624-201835040', '201816624-201835050', '201816624-201837218', '201816624-201845605', '201816642-201821178', '201816642-201822608', '201816642-201827252', '201816718-201817042', '201816718-201818254', '201816718-201872635', '201817042-201821444', '201817042-201866387', '201817046-201818254', '201817046-201830842', '201817046-201837010', '201817046-201840856', '201817046-201841526', '201817046-201877386', '201818254-201823988', '201818254-201835040', '201818254-201837228', '201818254-201846631', '201818346-201821444', '201818346-201821600', '201818492-201832446', '201818492-201835436', '201818492-201866387', '201818492-201871472', '201819010-201824030', '201819010-201845629', '201819712-201827252', '201819712-201841526', '201820961-201818254', '201820961-201819712', '201820961-201827252', '201820961-201827856', '201820961-201835040', '201820961-201837650', '201820961-201846343', '201820961-201877386', '201821178-201824030', '201821178-201825814', '201821178-201845629', '201821266-201821626', '201821266-201877386', '201821408-201822004', '201821444-201821594', '201821444-201822608', '201821444-201829236', '201821444-201836704', '201821444-201877386', '201821458-201821626', '201821520-201837650', '201821574-201825814', '201821594-201835050', '201821594-201839137', '201821594-201846625', '201821594-201846631', '201821600-201824030', '201821600-201837650', '201821626-201821738', '201821626-201824030', '201821626-201837228', '201821738-201839137', '201822004-201838048', '201822004-201845605', '201822608-201824244', '201822608-201835220', '201822608-201846631', '201823556-201837650', '201823988-201834588', '201824030-201837228', '201827252-201835436', '201827252-201836160', '201827252-201840856', '201827252-201845605', '201827856-201840856', '201829236-201834530', '201829236-201834666', '201829236-201841526', '201829236-201843006', '201830842-201840856', '201834508-201837650', '201834508-201843006', '201834508-201845973', '201834508-201846625', '201834530-201835748', '201834530-201837228', '201834530-201838048', '201834530-201845629', '201834588-201836160', '201834588-201836704', '201834588-201846631', '201835040-201845629', '201835220-201846625', '201835748-201842988', '201836704-201837650', '201836704-201846013', '201837010-201836160', '201837228-201840856', '201837228-201845629', '201837650-201846631', '201838074-201840856', '201841526-201845605', '201841526-201845973', '201842988-201843006', '201843006-201872635', '201845973-201871472', '20184625-20188366', '20184625-20188468', '201846343-201871472', '20184759-20186460', '201847953-201813821', '201847953-201821600', '201847953-201866311', '20184825-201846343', '20184825-20187260', '20184825-20188366', '20184825-20188910', '201855675-201861630', '201855675-201862744', '201855675-201865756', '201855859-201856087', '201856087-201859387', '201856087-201862744', '201856153-201856935', '201856219-201857277', '201856219-201864482', '201856229-201869746', '201858578-201864482', '201858578-201864722', '201859387-201864482', '201864482-201865574', '201864482-201875354', '20186460-201812237', '20186460-201812605', '20186460-201813506', '20186460-201813821', '20186460-201846343', '201864722-201862744', '201864722-201862914', '201864722-201865574', '201864722-201865756', '201865756-201875354', '20186746-201871472', '20186746-201874528', '20186746-20188468', '20187260-201812259', '20187260-201813821', '201872635-201874434', '20187456-201812605', '20187456-201816718', '20187456-201821444', '20187456-201829236', '20187456-201870893', '20187456-201871472', '20187456-20188016', '20187456-20189036', '201877386-201838048', '20188016-201871472', '20188016-20188468', '20188412-20189036', '20188468-201874434', '20189036-201812237', '20189036-201812605', '20189036-201813506', '20189036-201829236', '20189036-201872635', '20189038-201846343', '20189682-201872635', '20189946-201813536', '20189946-201813821']
# some arbritatry event pairs
EVID_PAIRS = ['201816096-201818492', '201823346-201837650', '20185009-20188468', '201815372-201819010', '201818254-201837218', '201856087-201865574', '201835040-201838074']
# for pub plot
EVID_PAIRS = ['20185009-20188468', '201816096-201818492']

if __name__ == '__main__':
    events = read_events('data/catalog_2018swarm.pha').filter(f'magnitude > {MAG}')
    import pandas as pd
    sta = pd.read_csv('data/station_coordinates.txt', sep='\s+', usecols=(0, 1, 2))
    coords = {s.station: (s.lat, s.lon) for idx, s in sta.iterrows()}
    events = filter_events(events, QU)
#    events = [ev for ev in events if str(ev.resource_id).split('/')[-1] in SELECT_IDS]
#    select_ids = {id_ for evpair in EVID_PAIRS for id_ in evpair.split('-')}
#    events = [ev for ev in events if str(ev.resource_id).split('/')[-1] in select_ids]
#    name_data_out = OUT + f'{YEAR}_mag>{MAG}_q{QU}_{len(events)}events'
    stream = create_stream(events, coords, write=False, name_data_out=None, alldata=False)
    print(len(stream), len(stream)/246/3)
#    if SELECT_IDS:
#        stream.traces = [tr for tr in stream if tr.meta.evid in SELECT_IDS]
    plot_xcorrs_pub(stream, tw='Scoda', timenorm='envelope')
    EVID_PAIRS = None
    ccs, ccs_stack = run_xcorr(stream, tw='Scoda', timenorm='envelope', write=True, write_all=False)
    ccs2, ccs_stack2 = run_xcorr(stream, tw='S', timenorm='envelope', write=True, write_all=False)
#    plot_xcorrs(stream, tw='Scoda', timenorm='envelope')

#    from IPython import start_ipython
#    start_ipython(user_ns=dict(ccs=ccs, ccs2=ccs2))
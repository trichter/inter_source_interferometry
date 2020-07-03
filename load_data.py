# Copyright 2020 Tom Eulenfeld, MIT license

"""
Recreate time window file if necessary, create plots of envelopes of all events
"""

import collections
import json
import pickle
import os.path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from obspy import read
import obspy
from tqdm import tqdm

from util.events import events2lists, event2list
from util.signal import envelope, smooth, get_local_minimum
from util.misc import collapse_json


('KVCN', 'P')
('NB', 'P')
('WERN', 'P')
('ROHR', 'P')
('KAC', 'P')
('GUNZ', 'P')
('MULD', 'P')
('TANN', 'P')
('LAEN', 'P')
('WERD', 'P')
('TRIB', 'P')
('VIEL', 'P')
('KVCN', 'S')
('NB', 'S')
('WERN', 'S')
('ROHR', 'S')
('KAC', 'S')

STATIONS = "NKC NKCN LBC KOPD VAC KVC KVCN STC POC SKC KRC NB WERN ROHR KAC MAC KOC GUNZ MULD TANN HUC LAEN ZHC TRC WERD TRIB LAC VIEL".split()
STATIONS = "NKC LBC VAC KVC STC POC SKC KRC WERN ROHR MAC KOC GUNZ MULD TANN HUC LAEN ZHC TRC WERD TRIB LAC VIEL".split()
STATIONS = "NKC LBC VAC KVC STC POC SKC KRC WERN ROHR MAC KOC GUNZ MULD TANN HUC LAEN ZHC TRC WERD TRIB LAC VIEL MANZ MSBB MROB MKON MGBB PLN TANN ROHR WERN".split()
MAN_PICKS = dict(P={'NKC': 1.6, 'LBC': 1.8, 'WERN': 2, 'VAC': 2, 'ROHR': 2.2, 'KVC': 2, 'GUNZ': 3, 'MULD': 3.2, 'TANN': 3.2, 'LAEN': 3.7, 'ZHC': 4.2, 'WERD': 4.2, 'TRIB': 4.2, 'VIEL': 4.5, 'MANZ': 5, 'MSBB': 5, 'MROB': 5, 'MKON': 5, 'MGBB': 5, 'PLN': 5},
                 S={'NKC': 3, 'LBC': 3, 'WERN': 3.5,'ROHR': 4, 'KVC': 3.8, 'GUNZ': 5.5, 'MULD': 6, 'TANN': 6, 'LAEN': 6.5, 'ZHC': 7, 'WERD': 7.5, 'TRIB': 7.6, 'VIEL': 7.6, 'MANZ': 8, 'MSBB': 8, 'MROB': 8, 'MKON': 8, 'MGBB': 8, 'PLN': 8})

YEAR = '2018'
MAG = 1.8
EVENT_FILE_ALL = 'data/catalog_2018swarm.pha'
PKL_EVENT_FILE_ALL = f'tmp/cat_{YEAR}.pkl'

DATA_FILES1 = 'data/waveforms/{id}_{sta}_?H?.mseed' # WEBNET
#DATA_FILES2 = 'data_2018_other/{id}_*/*.mseed'  # BGR, LMU

QCFILE = 'data2/2018_events_qc_mag1.9.txt'
TWFILE = 'data2/2018_events_tw_mag1.9.txt'
TXTFILE = 'tmp/cat_2018_mag>1.8.txt'
LATLON0 = (50.25, 12.45)


def get_events():
    from obspy import read_events
    if not os.path.exists(PKL_EVENT_FILE_ALL):
        events = read_events(EVENT_FILE_ALL)
        with open(PKL_EVENT_FILE_ALL, 'wb') as f:
            pickle.dump(events, f, pickle.HIGHEST_PROTOCOL)
    with open(PKL_EVENT_FILE_ALL, 'rb') as f:
        return pickle.load(f)


def create_event_qc_list(events):
    events = sorted(events, key=lambda ev: ev.origins[0].time)
    ids, _, _, _, _, mags, _ = zip(*events2lists(events))
    out = ' \n'.join(i + ' M' + str(m) for i, m in zip(ids, mags))
    with open('tmp/2018_evt_list.txt', 'w') as f:
        f.write(out)


def get_picks(events, manipulate=True):
    def _pkey(seedid):
        sta, cha = seedid.split('.')[1::2]
        return (sta, 'P' if cha[-1] == 'Z' else 'S' if cha[-1] == 'N' else None)
    events = sorted(events, key=lambda ev: ev.origins[0].time)
    ids, otimes, _, _, _, _, picks = zip(*events2lists(events))
    relpicks = collections.defaultdict(list)
    for otime, event_picks in zip(otimes, picks):
        for seedid, phase, ptime in event_picks:
            assert _pkey(seedid)[1] == phase
            relpicks[_pkey(seedid)].append(ptime - otime)
    relpicks_mean = {k: (np.mean(v), 'mean') for k, v in relpicks.items()}
    for phase in 'PS':
        for sta, v in MAN_PICKS[phase].items():
            if (sta, phase) not in relpicks:
                relpicks_mean[(sta, phase)] = (v, 'manual')
    allpicks = {}
    relpicks = {}
    for id_, otime, event_picks in zip(ids, otimes, picks):
        abs_event_picks = {_pkey(seedid): (ptime, 'pick') for seedid, phase, ptime in event_picks}
        rel_event_picks = {_pkey(seedid): (ptime - otime, 'pick') for seedid, phase, ptime in event_picks}
        for phase in 'PS':
            for sta in STATIONS:
                if (sta, phase) not in abs_event_picks:
                    relpick = relpicks_mean[(sta, phase)]
                    abs_event_picks[(sta, phase)] = (otime + relpick[0], relpick[1])
                    rel_event_picks[(sta, phase)] = relpick
        allpicks[id_] = abs_event_picks
        relpicks[id_] = rel_event_picks
    ev2ev = {id1: tuple(id2 for id2, time2 in zip(ids, otimes) if id1 != id2 and -20 < time2 - time1 < 50) for id1, time1 in zip(ids, otimes)}
    return allpicks, relpicks, relpicks_mean, ev2ev


def get_tw(tr, otime, spickrel):
#    i1 = np.argmax(tr.slice(otime, otime + 10).data)
    t1 = otime + spickrel + 1
    noise = 1.2 * np.percentile(tr.slice(None, otime + 1).data, 5)
    try:
        i2 = np.where(tr.slice(t1, None).data < noise)[0][0]
    except:
        t2 = tr.stats.endtime
    else:
        t2 = t1 + i2 * tr.stats.delta
    t3 = get_local_minimum(tr.slice(t1, t2), ratio=10, seconds_before_max=2.5)
    if t3 is not None:
        t3 = t3 - otime
    return t1 - otime, t2 - otime, t3, noise


def iter_data(events, alldata=False):
    for event in tqdm(events):
        id_, *_ = event2list(event)
        try:
            stream = read(DATA_FILES1.format(id=id_, sta='*'))
        except Exception as ex:
            print(ex, id_)
            continue
        if alldata:
            stream2 = read(DATA_FILES2.format(id=id_))
            stream += stream2
        yield stream, event


def get_envelope(tr):
    tr = tr.copy()
    tr.data = envelope(tr.data)
    tr.data = smooth(tr.data, int(round(0.2 * tr.stats.sampling_rate)))
    return tr


def single_plot(stream, event, tw=None):
    id_, otime, lon, lat, dep, mag, _ = event2list(event)
    stream.filter('highpass', freq=5).trim(otime-20, otime + 60)
    stream.traces = sorted(stream.traces, key=lambda tr: (STATIONS.index(tr.stats.station), tr.stats.channel))
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    if tw is None:
        ax.annotate(f'{id_} M{mag:.1f}', (0.05, 0.95), xycoords='axes fraction', va='top')
    else:
        ax.annotate(f'{id_} M{mag:.1f} {tw[id_][1]}', (0.05, 0.95), xycoords='axes fraction', va='top')

    ax.axvline(0, color='C0')
    ax.axvline(10, color='0.8')
    ax.axvline(40, color='0.8')
    for i, tr in enumerate(stream):
        ax.annotate(tr.id, (49, i), ha='right')
        try:
            scale = 1 / np.max(tr.slice(otime + 2, otime + 15).data)
        except ValueError as ex:
            print(str(ex))
            continue
        # plot data and envelope
        trenv = get_envelope(tr)
        tr.trim(otime-10, otime + 50)
        trenv.trim(otime-10, otime + 50)
        ax.plot(tr.times(reftime=otime), i + scale * tr.data, color='0.6', lw=1)
        ax.plot(trenv.times(reftime=otime), i + scale * trenv.data, color='C0', lw=1)
        # plot picks
        sta = tr.stats.station
        ppick = RELPICKS[id_][(sta, 'P')][0]
        spick = RELPICKS[id_][(sta, 'S')][0]
        ax.vlines([ppick, spick], i-0.25, i + 0.25, color='C1', zorder=10)
        p_ = [RELPICKS[id2][sta, phase][0] for id2 in EV2EV[id_] for phase in 'PS']
        p_ = [p for p in p_ if -10 < p < 50]
        ax.vlines(p_, i-0.25, i + 0.25, color='C2', zorder=10)
        if tw is None:
            # calc and plot time windows
            t1, t2, t3, _ = get_tw(trenv, otime, spick)
            if t3 is not None:
                rect = Rectangle((t2, i-0.4), t3-t2, 0.8, alpha=0.1, facecolor='C1')
                ax.add_patch(rect)
            rect = Rectangle((t1, i-0.4), t2-t1, 0.8, alpha=0.2, facecolor='C0')
            ax.add_patch(rect)
        else:
            try:
                t1, t2 = tw[id_][2][stacomp(tr.id)][:2]
            except KeyError:
                pass
            else:
                rect = Rectangle((t1, i-0.4), t2-t1, 0.8, alpha=0.3, facecolor='C0')
                ax.add_patch(rect)
#            noise = tw[id_][2][stacomp(tr.id)][3]
#            ax.axhline(i + scale * noise, color='0.6')
    ax.set_xlabel('time (s)')
    ax.set_ylim(-2, len(stream) + 2)
    ax.set_xticks([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xticks(range(-10, 51), minor=True)
    ax.grid(which='major', axis='x')
    ax.grid(which='minor', axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelleft=False, left=False)
    if tw is None:
        plt.savefig(f'tmp/{otime!s:.21}_{id_}_M{mag:.1f}.png', bbox_inches='tight')
    else:
        plt.savefig(f'tmp/{otime!s:.21}_{id_}_M{mag:.1f}_{tw[id_][1]}.png', bbox_inches='tight')
    plt.close()


def stacomp(seedid):
    sta, cha = seedid.split('.')[1::2]
    return sta + '.' + cha[-1]


def tw_from_qc_file(events=None):
    if os.path.exists(TWFILE):
        with open(TWFILE) as f:
            return json.load(f)
    print('Need to rebuild qc file ...')
    assert events
    with open(QCFILE) as f:
        text = f.read()
    qc = {line.split()[0]: line.split(maxsplit=2)[2] for line in text.splitlines() if line.startswith('2018')}
    tw = {}
    for stream, event in iter_data(events):
        id_, otime, lon, lat, dep, mag, _ = event2list(event)
        trtw = {}
        stream.filter('highpass', freq=5).trim(otime-20, otime + 60)
        for tr in stream:
            sta = tr.stats.station
            spick = RELPICKS[id_][(sta, 'S')][0]
            ppick = RELPICKS[id_][(sta, 'P')][0]
            t1 = spick + 1
            t2 = 50; r = 'max_tw'
            p_ = [RELPICKS[id2][sta, phase][0] for id2 in EV2EV[id_] for phase in 'PS']
            p_ = [p for p in p_ if spick + 1 < p < 50]
            if len(p_) > 0:
                t3 = sorted(p_)[0]
                if t3 < t2:
                    t2 = t3
                    r = 'next_pick'
            trenv = get_envelope(tr).trim(otime - 15, otime + 55)
            try:
                _, t3, t4, noise = get_tw(trenv, otime, spick)
                assert round(t1 - _, 5) == 0
            except IndexError as ex:
                print('error in calculating noise', id_, tr.id, str(ex))
            else:
                if t4 is None and t3 < t2:
                    t2 = t3
                    r = 'noise_level_reached'
                elif t4 is not None and t4 < t2:
                    if r == 'max_tw' or t4+2.4 < t2:
                        t2 = t4
                        r = 'raising_envelope'
            fields = qc[id_].split()
            if 's' in qc[id_]:
                t3 = ppick + [float(f[:-1]) for f in fields if f.endswith('s')][0]
                if t3 < t2:
                    t2 = t3
                    r = 'visually_inspected'
            trtw[stacomp(tr.id)] = (round(t1, 2), round(t2, 2), r, round(noise, 2))
        qdict = {'xx': 0, 'x': 0, 'Q1': 3, 'Q2': 2, 'Q3': 1}
        q = qdict[[f for f in fields if f[0] in 'Qx'][0]]
        tw[id_] = [q, qc[id_], trtw]
    text = collapse_json(json.dumps(tw, indent=2), 6)
    with open(TWFILE, 'w') as f:
        f.write(text)
    return tw


def ev2id(event):
    return str(event.resource_id).split('/')[-1]


def select_events(events, tw, qclevel):
    events = [ev for ev in events if tw.get(ev2id(ev), (0,))[0] >= qclevel]
    events = [ev for ev in events if 'NOWEBNET' not in tw[ev2id(ev)][1]]
    return obspy.Catalog(events)


if __name__ == '__main__':
    print('load events')
    allevents = get_events()
    print('finished loading')
    allevents = obspy.Catalog(sorted(allevents, key=lambda ev: ev.origins[0].time))
    PICKS, RELPICKS, _, EV2EV = get_picks(allevents)
    events = allevents.filter(f'magnitude > {MAG}')
#    create_event_qc_list(events)
    tw = tw_from_qc_file(events)
    for stream, event in iter_data(events, alldata=False):
        single_plot(stream, event, tw=tw)

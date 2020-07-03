# Copyright 2020 Tom Eulenfeld, MIT license

def get_picks(picks, arrivals):
    """Picks for specific station from arrivals"""
    try:
        ps = []
        for arrival in arrivals:
            phase = arrival.phase
            p = arrival.pick_id.get_referred_object()
            if p is None:
                print('DID NOT FIND PICK ID')
                raise ValueError
            seedid = p.waveform_id.get_seed_string()
            ps.append((seedid, phase, p.time))
        return ps
    except ValueError:
        pass
    ps = []
    for p in picks:
        seedid = p.waveform_id.get_seed_string()
        phase = p.phase_hint
        ps.append((seedid, phase, p.time))
    return ps


def event2dict(ev):
    id_ = str(ev.resource_id).split('/')[-1]
    ori = ev.preferred_origin() or ev.origins[0]
    mag = ev.preferred_magnitude() or ev.magnitudes[0]
    try:
        mag = mag.mag
    except:
        mag = None
    try:
        picks = get_picks(ev.picks, ori.arrivals)
    except Exception as ex:
        print(ex)
        picks = None

    return dict(
            id=id_, time=ori.time,
            lat=ori.latitude, lon=ori.longitude, dep=ori.depth / 1000,
            mag=mag, picks=picks)
    pass


def event2list(ev):
    """id, time, lon, lat, dep_km, mag, picks"""
    d = event2dict(ev)
    return (d['id'], d['time'], d['lon'], d['lat'], d['dep'], d['mag'],
            d['picks'])


def events2dicts(events):
    return [event2dict(ev) for ev in events]


def events2lists(events):
    """id, time, lon, lat, dep_km, mag, picks"""
    return [event2list(ev) for ev in events]


def load_webnet(year='*', plot=False, stations=False):
    import glob
    import pandas as pd

    names = ['time', 'lon', 'lat', 'dep', 'mag']
    kwargs = dict(sep=';', skipinitialspace=True, skiprows=3, parse_dates=[0],
                  names=names)
    path =f'data/webnet/*.txt'
    frames = [pd.read_csv(fname, **kwargs) for fname in glob.glob(path)]
    if len(frames) == 0:
        print('You can obtain the WEBNET catalog at ig.cas.cz. Put the txt files in new webnet directory inside data.')
        eqs = None
    else:
        eqs = pd.concat(frames, ignore_index=True)
    if stations:
        path = 'data/station_coordinates.txt'
        sta = pd.read_csv(path, sep='\s+', usecols=(0, 1, 2))
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.scatter(eqs.lon.values, eqs.lat.values, 1, eqs.time.values)
        ax2.scatter(eqs.time.values, eqs.mag.values, 1, eqs.time.values)
        if stations:
            ax1.scatter(sta.lon.values, sta.lat.values, 100, marker='v',
                        color='none', edgecolors='k')
            for idx, s in sta.iterrows():
                ax1.annotate(s.station, (s.lon, s.lat), (5, 5),
                             textcoords='offset points')
        plt.show()
    if stations:
        return eqs, sta
    return eqs

# Copyright 2020 Tom Eulenfeld, MIT license

import numpy as np
import scipy.signal


def _seconds2utc(self, seconds, reftime=None):  # same as in rf package
    """Return UTCDateTime given as seconds relative to reftime"""
    from collections import Iterable
    from obspy import UTCDateTime as UTC
    if isinstance(seconds, Iterable):
        return [_seconds2utc(self, s, reftime=reftime) for s in seconds]
    if isinstance(seconds, UTC) or reftime is None or seconds is None:
        return seconds
    if not isinstance(reftime, UTC):
        reftime = self.stats[reftime]
    return reftime + seconds


def trim2(self, starttime=None, endtime=None, reftime=None, check_npts=True, **kwargs):
    # same as in rf package + mid possibility
    """
    Alternative trim method accepting relative times.

    See :meth:`~obspy.core.stream.Stream.trim`.

    :param starttime,endtime: accept UTCDateTime or seconds relative to
        reftime
    :param reftime: reference time, can be an UTCDateTime object or a
        string. The string will be looked up in the stats dictionary
        (e.g. 'starttime', 'endtime', 'onset').
    """
    for tr in self.traces:
        st = tr.stats
        ref = (st.starttime + 0.5 * (st.endtime - st.starttime)
               if reftime in ('mid', 'middle') else reftime)
        t1 = _seconds2utc(tr, starttime, reftime=ref)
        t2 = _seconds2utc(tr, endtime, reftime=ref)
        tr.trim(t1, t2, **kwargs)
    if check_npts:
        npts = int(round(np.median([len(tr) for tr in self.traces])))
        self.traces = [tr for tr in self.traces if len(tr) >= npts]
        for tr in self.traces:
            tr.data = tr.data[:npts]
    return self


def smooth(x, window_len, window='flat', method='zeros'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an
        odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    :param method: handling of border effects 'zeros', 'reflect', None
        'zeros': zero padding on both ends (len(smooth(x)) = len(x))
        'reflect': pad reflected signal on both ends (same)
        None: no handling of border effects
            (len(smooth(x)) = len(x) - len(window_len) + 1)

    See also:
    www.scipy.org/Cookbook/SignalSmooth
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 2:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if method == 'zeros':
        s = np.r_[np.zeros((window_len - 1) // 2), x,
                  np.zeros(window_len // 2)]
    elif method == 'reflect':
        s = np.r_[x[(window_len - 1) // 2:0:-1], x,
                  x[-1:-(window_len + 1) // 2:-1]]
    else:
        s = x
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    return np.convolve(w / w.sum(), s, mode='valid')


def envelope(data):
    from scipy.signal import hilbert
    from scipy.fftpack import next_fast_len
    nfft = next_fast_len(len(data))
    anal_sig = hilbert(data, N=nfft)[:len(data)]
    return np.abs(anal_sig)


def get_local_minimum(tr, smooth=None, ratio=5, smooth_window='flat', seconds_before_max=None):
    """
    tr: Trace
    smooth: bool
    ratio: ratio of local minima to maxima
    smooth_window

    """
    data = tr.data
    if smooth:
        window_len = int(round(smooth * tr.stats.sampling_rate))
        try:
            data = smooth(tr.data, window_len=window_len, method='clip',
                           window=smooth_window)
        except ValueError:
            pass
    mins = scipy.signal.argrelmin(data)[0]
    maxs = scipy.signal.argrelmax(data)[0]
    if len(mins) == 0 or len(maxs) == 0:
        return
    mins2 = [mins[0]]
    for mi in mins[1:]:
        if data[mi] < data[mins2[-1]]:
            mins2.append(mi)
    mins = np.array(mins2)
    for ma in maxs:
        try:
            mi = np.nonzero(mins < ma)[0][-1]
            mi = mins[mi]
        except IndexError:
            mi = 0
        if data[ma] / data[mi] > ratio:
            if seconds_before_max is not None:
                mi = max(mi, ma - seconds_before_max / tr.stats.delta)
            return tr.stats.starttime + mi * tr.stats.delta
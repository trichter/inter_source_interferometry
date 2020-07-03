# Copyright 2020 Tom Eulenfeld, MIT license

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi
from obspy.imaging.scripts.mopad import strikediprake_2_moments, NED2USE, MomentTensor
from obspy.core.event.source import farfield as obsfarfield
from obspy.imaging.beachball import beach


def _full_mt(mt):
    return np.array([[mt[0], mt[3], mt[4]],
                     [mt[3], mt[1], mt[5]],
                     [mt[4], mt[5], mt[2]]])


def _cart2sph_pnt(x, y, z):
    ind = z < 0
    z[ind] *= -1
    x[ind] *= -1
    y[ind] *= -1
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = pi/2 - np.arctan2(z, hxy)  # inclination meassured between z and r
    az = (pi / 2 - np.arctan2(x, y) ) % (2 * pi)
    assert np.all(el <= pi/2)
    assert np.all(np.abs(r-1)<0.001)
    return r, el, az


def pnt_axis(mt, out='xyz'):
    # PNT axis are in the coordinate system in which mt is defined
    # for out='rad', system must be NED (north, east, down)
    # returns T, N and P axis
    evals, evecs = np.linalg.eigh(_full_mt(mt))
    assert evals[0] <= evals[1] <= evals[2]
    if np.linalg.det(evecs) < 0:
        evecs *= -1
    if out == 'rad':
        evecs = np.transpose(_cart2sph_pnt(*evecs)[1:])
    return evecs


def _polar_unitvecs(theta, phi):
    unitvec = [[sin(theta) * cos(phi), sin(theta) * sin(phi), +cos(theta)],
               [cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)],
               [-sin(phi), cos(phi), 0 * phi]]
    return np.array(unitvec)


def farfield(mt, points, out='xyz', typ='P'):
    # mt is strike dip rake or MT in NED convention (North East Down)
    if len(mt) == 3:
        mt = strikediprake_2_moments(*mt)
    ff = obsfarfield(mt, np.array(points), typ)
    if out == 'rad':
        # assume points are given in polar projection
        # project xyz to r, theta, phi unit vectors
        ffn = np.einsum('ij,kij->kj', ff, _polar_unitvecs(*points))
    return ffn


def plot_pnt(mt, ax=None, what='PT', color='k', label='PNT', zorder=None):
    # mt is strike dip rake or MT in NED convention (North East Down)
    if len(mt) == 3:
        mt = strikediprake_2_moments(*mt)
    pnt = pnt_axis(mt, out='rad')
    xy1 = pnt[0, 1], np.rad2deg(pnt[0, 0])
    xy2 = pnt[2, 1], np.rad2deg(pnt[2, 0])
    xy3 = pnt[1, 1], np.rad2deg(pnt[1, 0])
    lkw = dict(fontsize='large', xytext=(3, 1), textcoords='offset points')
    if ax is None:
        ax = plt.subplot(111)
    if 'P' in what:
        ax.scatter(*xy1, color=color, marker='.', zorder=zorder)
        if 'P' in label:
            ax.annotate('P', xy1, **lkw)
    if 'T' in what:
        ax.scatter(*xy2, color=color, marker='.', zorder=zorder)
        if 'T' in label:
            ax.annotate('T', xy2, **lkw)
    if 'N' in what:
        ax.scatter(*xy3, color=color, marker='.', zorder=zorder)
        if 'N' in label:
            ax.annotate('N', xy3, **lkw)


def plot_farfield(mt, typ='P', points=None, thetalim=(0, 90), ax=None,
                  scale=None, plot_axis=None):
    # mt is strike dip rake or MT in NED convention (North East Down)
    if len(mt) == 3:
        mt = strikediprake_2_moments(*mt)
    mt_use = NED2USE(mt)
    if typ in ('P', 'S'):
        if points is None:
            theta = np.linspace(0.1, pi/2, 6)
            phi = np.linspace(0, 2*pi, 11, endpoint=False)[:, np.newaxis]
            phi2, theta2 = np.meshgrid(phi, theta)
            phi2 = np.ravel(phi2)
            theta2 = np.ravel(theta2)
        else:
            phi2, theta2 = np.array(points)
        ff = farfield(mt, [theta2, phi2], out='rad', typ=typ)
    if ax is None:
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
    else:
        fig = ax.figure
        ax2 = ax
    ax2.set_aspect(1)
    ax2.axison=False
    b = beach(mt_use, width=20, edgecolor='k', nofill=True, linewidth=1)
    ax2.add_collection(b)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)

#    ax1 = fig.add_subplot(111, polar=True)
    ax1 = fig.add_axes(ax2.get_position().bounds, polar=True)
    ax1.grid(False)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    if typ == 'P':
        assert np.sum(np.abs(ff[1:, :])) < 1e-4
        ff = ff[0, :]
        ind1 = ff > 0
        ind2 = ff < 0
        ax1.scatter(phi2[ind1], np.rad2deg(theta2[ind1]), (20 * ff[ind1]) ** 2, marker='+')
        ax1.scatter(phi2[ind2], np.rad2deg(theta2[ind2]), (20 * ff[ind2]) ** 2, marker='o')
    elif typ == 'S':
        assert np.sum(np.abs(ff[0, :])) < 1e-4
        dr, dphi = ff[1, :], ff[2, :]
        # convert to xy
        dx = dr * cos(phi2) - dphi * sin(phi2)
        dy = dr * sin(phi2) + dphi * cos(phi2)
        #dx, dy -> 90° drehen: -dy,dx -> phi andersherum dy, dx
        #zusaetzliches minuszeichen ist ein bisschen strange
        ax1.quiver(phi2, np.rad2deg(theta2), -dy, -dx, pivot='mid', scale=scale)

    ax1.set_ylim(thetalim)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axison=False
    if plot_axis:
        plot_pnt(mt, ax1, plot_axis)
    return ax1, ax2


#    def onclick(event):
#        xy = np.rad2deg(event.xdata) % 360, event.ydata
#        print('({:6.2f}, {:5.2f}),'.format(*xy))
#    fig.canvas.mpl_connect('button_press_event', onclick)


def nice_points():
    points=[
    (281.86, 81.98),  # links rechts, nodel plane
    (294.29, 61.45),
    (311.35, 43.50),
    (  0.00, 28.93),
    ( 36.37, 33.55),
    ( 62.79, 50.09),
#    ( 78.09, 71.16),
    ( 76, 68),
#    ( 87.88, 86.99),
    (84, 82),

    (352.32, 85.74),  # oben unten, nodal plane
    (347.23, 68.49),
    (341.10, 49.62),
    #(316.53, 22.32),
    #(230.05, 17.21),
    #(191.14, 44.76),
    #(181.52, 73.36),
    #(177.57, 86.81),

    (  0.00, 46.95),  # links rechts, -1
    ( 27.21, 50.05),
    ( 54.57, 63.35),
    ( 68.91, 80.39),
    (326.02, 58.38),
    (305.98, 71.62),
    (295.68, 87.00),

    (323.48, 82.66),  # links rechts, -2
    (336.13, 75.83),
    (  6.23, 67.31),
    ( 27.19, 71.40),
    ( 47.72, 84.26),

    (  9.33, 84.10),  # links rechts, -3

    (264.14, 74.41),  # links rechts, 1
    (268.57, 50.67),
    (277.72, 30.47),
    (290.81, 15.63),
    ( 58.34,  6.87),
    ( 96.08, 30.37),
    (103.75, 60.67),
#    (108.72, 85.88),
    (110, 81),

    (244.08, 73.10),  # links rechts, 2
    (245.39, 49.82),
    (233, 36),
#    (231.62, 31.06),
    (211.40, 24.31),
    (161.01, 22.46),
    (131.78, 39.18),
    (126.90, 66.99),
    (129.14, 82.25),

    (226.38, 77.37),  # links rechts, 3
    (214.62, 56.58),
    (190.12, 44.33),
    (162.79, 47.72),
    (147.40, 65.09),
    (145.15, 85.22),

    (206.63, 77.14),  # links rechts, 4
    (183.77, 66.67),
    (162.60, 73.29),

    (178.66, 83.59),  # ganz unten
    ]

    return np.transpose(np.deg2rad(points))


def test():
    sdr = [175, 75, -30]
    plot_farfield(sdr, points=nice_points(), plot_pt=True)
    plot_farfield(sdr, typ='S', points=nice_points(), plot_pt=True)

    mt = strikediprake_2_moments(*sdr)  # NED
    obj = MomentTensor(mt)
    obj._M_to_principal_axis_system()
    print(obj.get_fps())
    print(obj.get_p_axis(style='f'))
    print(obj.get_t_axis(style='f'))
    print(obj.get_null_axis(style='f'))
    print(pnt_axis(mt, out='xyz'))
    print(pnt_axis(mt, out='rad'))
    plt.show()


#test()
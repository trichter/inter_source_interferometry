# Copyright 2020 Tom Eulenfeld, MIT license

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np


# https://epsg.io/3035
EA_EURO = ccrs.LambertAzimuthalEqualArea(
    central_longitude=10, central_latitude=52,
    false_easting=4321000, false_northing=3210000,
    globe=ccrs.Globe(ellipse='GRS80'))

GEO = ccrs.Geodetic()

def add_scale(ax, length, loc, crs=GEO, lw=1,
              cap=2, label=True, size=None, vpad=2):
    bx, by = ax.projection.transform_point(loc[0], loc[1], crs)
    bx1, bx2 = bx - 500 * length, bx + 500 * length
    ax.plot((bx1, bx2), (by, by), color='k', linewidth=lw)
    if cap:
        kw = {'xycoords': 'data', 'textcoords': 'offset points', 'arrowprops':
              {'arrowstyle': '-', 'connectionstyle': 'arc3',
               'shrinkA': 0, 'shrinkB': 0, 'linewidth': lw}}
        ax.annotate('', (bx1, by), (0, cap), **kw)
        ax.annotate('', (bx1, by), (0, -cap), **kw)
        ax.annotate('', (bx2, by), (0, cap), **kw)
        ax.annotate('', (bx2, by), (0, -cap), **kw)
    if label:
        ax.annotate(str(length) + ' km', (bx, by), (0, vpad), size=size,
                    textcoords='offset points', ha='center', va='bottom')


def de_border(ax, edgecolor='k', facecolor='none', **kw):
    fname = '/home/eule/data/geo/de/border/Germany_AL2-AL2.shp'
    germany = shpreader.Reader(fname)
    ax.add_geometries(germany.geometries(), GEO, facecolor=facecolor,
                      edgecolor=edgecolor, **kw)


def add_ticklabels(ax, lonticks, latticks, fmt='%.1f°'):
    if hasattr(ax, 'projection'):
        xticks, _, _ = zip(*ax.projection.transform_points(
                GEO, np.array(lonticks), np.ones(len(lonticks)) * latticks[0]))
        _, yticks, _ = zip(*ax.projection.transform_points(
                GEO, np.ones(len(latticks)) * lonticks[0], np.array(latticks)))
    else:
        xticks = lonticks
        yticks = latticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(fmt % abs(l) + 'E' if l> 0 else 'W' for l in lonticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(fmt % abs(l) + 'N' if l > 0 else 'S' for l in latticks)


def get_elevation_srtm(lon1, lat1, dx=1, dy=1, max_distance=10,
                       shading=True, azimuth=315, altitude=45):
    # https://github.com/SciTools/cartopy/issues/789
    from http.cookiejar import CookieJar
    import urllib.request
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", 'your_user_name', 'your_password')
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(password_manager),
        urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)
    # end patch
    from cartopy.io.srtm import SRTM1Source, add_shading
    elev, crs, extent = SRTM1Source().combined(lon1, lat1, dx, dy)
    shades = None
    if shading:
        shades = add_shading(elev, azimuth, altitude)
    return elev, shades, crs, extent


def plot_elevation(ax, cmap='Greys', rasterized=False, **kw):
    x1, x2, y1, y2 = ax.get_extent()
    x, y = np.array([x1, x1, x2, x2]), np.array([y1, y2, y1, y2])
    lons, lats, _ = zip(*GEO.transform_points(ax.projection, x, y))
    lon, lat = int(np.min(lons)), int(np.min(lats))
    dx = int(np.ceil(np.max(lons) - lon))
    dy = int(np.ceil(np.max(lats) - lat))
    try:
        elev, shades, crs, extent = get_elevation_srtm(lon, lat, dx, dy, **kw)
    except Exception:
        import traceback
        print(traceback.format_exc())
        print('To download SRTM elevation you need to set up an account and add your credentials inside this module')
    else:
        if shades is not None:
            elev = shades
        ax.imshow(elev, extent=extent, transform=crs, cmap=cmap, origin='lower', rasterized=rasterized)


def convert_coords2km(coords, latlon0=None):
    import utm
    x, y = zip(*[utm.from_latlon(lat1, lon1)[:2]
                 for lat1, lon1, *_ in coords])
    if latlon0 is None:
        x0 = np.mean(x)
        y0 = np.mean(y)
    else:
        x0, y0 = utm.from_latlon(*latlon0)[:2]
    x = (np.array(x) - x0) / 1000
    y = (np.array(y) - y0) / 1000
    if len(coords[0]) == 3:
        return list(zip(x, y, [c[2] for c in coords]))
    else:
        return list(zip(x, y))
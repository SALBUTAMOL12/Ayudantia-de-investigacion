"""
One point per census zone for the imagery download (2017).

The microdata zone id (ID_ZONA_LOC) is sequential; its geography is (COMUNA, DC, AREA, ZC_LOC):
urban (AREA=1) -> zona censal of the manzana shapefile, rural (AREA=2) -> localidad of the entidad
shapefile. Rural households are identified only to the localidad, so the point is the localidad, not
each entidad (the RA's pilot used entidades and grouped urban zones by ZONA_CENSA within comuna,
which merges zones with the same number in different districts).

Point = household-weighted mean of manzana/entidad centroids (CANTIDAD_H; equal weights when the zone
has no households). Also reports how many of the zone's households fall inside a box of --box_m metres
around the point.

    python -m scripts.eo_ml.zone_points --year 2017

Output: data/B_inter/eo/2017/zone_points.parquet (+ .csv) with zone_id, comuna, dc, area, zc_loc,
lon, lat, n_hh, n_units, share_hh_in_box.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PROJ_DATA', str(Path(sys.prefix) / 'share' / 'proj'))

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'A_raw' / 'raw' / 'INE' / 'census_2017'
OUT = ROOT / 'data' / 'B_inter' / 'eo'
METRIC_CRS = 32719  # UTM 19S


def num(s):
    return pd.to_numeric(s, errors='coerce').fillna(0)


def units_2017():
    m = gpd.read_file(RAW / 'Microdatos_Manzana.shp', columns=['CUT', 'CODIGO_DIS', 'ZONA_CENSA', 'CANTIDAD_H'])
    m = gpd.GeoDataFrame(dict(comuna=num(m.CUT).astype(int), dc=num(m.CODIGO_DIS).astype(int), area=1,
                              zc_loc=num(m.ZONA_CENSA).astype(int), hh=num(m.CANTIDAD_H)), geometry=m.geometry)
    e = gpd.read_file(RAW / 'Microdatos_Entidad.shp', columns=['COD_COMUNA', 'DISTRITO', 'AREA', 'CODIGO_LOC', 'CANTIDAD_H'])
    e = gpd.GeoDataFrame(dict(comuna=num(e.COD_COMUNA).astype(int), dc=num(e.DISTRITO).astype(int),
                              area=num(e.AREA).astype(int), zc_loc=num(e.CODIGO_LOC).astype(int),
                              hh=num(e.CANTIDAD_H)), geometry=e.geometry)
    u = pd.concat([m.to_crs(METRIC_CRS), e.to_crs(METRIC_CRS)], ignore_index=True)
    u['x'], u['y'] = u.geometry.centroid.x, u.geometry.centroid.y
    return pd.DataFrame(u.drop(columns='geometry'))


def zone_ids_2017():
    v = pd.read_csv(RAW / 'Microdato_Censo2017-Viviendas.csv', sep=';',
                    usecols=['COMUNA', 'DC', 'AREA', 'ZC_LOC', 'ID_ZONA_LOC'])
    v = v.drop_duplicates('ID_ZONA_LOC').rename(columns=str.lower).rename(columns={'id_zona_loc': 'zone_id'})
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2017, choices=[2017])
    ap.add_argument('--box_m', type=float, default=2560, help='side of the image box in metres (256 px x 10 m)')
    a = ap.parse_args()

    key = ['comuna', 'dc', 'area', 'zc_loc']
    u = units_2017()
    w = np.where(u.groupby(key).hh.transform('sum') > 0, u.hh, 1.0)
    u['w'] = w
    g = u.assign(wx=u.x * w, wy=u.y * w).groupby(key)
    pts = g[['wx', 'wy', 'w']].sum()
    pts['x'], pts['y'] = pts.wx / pts.w, pts.wy / pts.w
    pts['n_hh'] = g.hh.sum()
    pts['n_units'] = g.size()
    pts = pts.reset_index()

    # share of households whose manzana/entidad centroid lies inside the box
    u = u.merge(pts[key + ['x', 'y']].rename(columns={'x': 'px', 'y': 'py'}), on=key)
    half = a.box_m / 2
    u['inside'] = ((u.x - u.px).abs() <= half) & ((u.y - u.py).abs() <= half)
    inside = u.assign(hh_in=u.hh * u.inside).groupby(key).hh_in.sum()
    pts = pts.merge(inside.rename('hh_in').reset_index(), on=key)
    pts['share_hh_in_box'] = np.where(pts.n_hh > 0, pts.hh_in / pts.n_hh.where(pts.n_hh > 0), np.nan)

    ll = gpd.GeoSeries(gpd.points_from_xy(pts.x, pts.y), crs=METRIC_CRS).to_crs(4326)
    pts['lon'], pts['lat'] = ll.x.values, ll.y.values

    z = zone_ids_2017().merge(pts, on=key, how='left')
    miss = z.lon.isna()
    print(f'zones in microdata: {len(z):,}; with a point: {(~miss).sum():,}; without: {miss.sum()} '
          f'(dc/zc_loc 99/999 etc.)')
    z = z.loc[~miss, ['zone_id'] + key + ['lon', 'lat', 'n_hh', 'n_units', 'share_hh_in_box']]
    z['zone_id'] = z.zone_id.astype(int)
    for ar, d in z.groupby('area'):
        sh = np.average(d.share_hh_in_box.fillna(0), weights=d.n_hh) if d.n_hh.sum() else np.nan
        print(f'area {ar}: {len(d):,} zones, households inside the {a.box_m:.0f} m box: {sh:.1%}')

    out = OUT / str(a.year)
    out.mkdir(parents=True, exist_ok=True)
    z.to_parquet(out / 'zone_points.parquet', index=False)
    z.to_csv(out / 'zone_points.csv', index=False)
    print('wrote', out / 'zone_points.parquet')


if __name__ == '__main__':
    main()

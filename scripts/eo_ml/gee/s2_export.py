"""
Sentinel-2 mosaics per batch of census zones, exported to Google Drive (2017 design).

Recipe (follows the RA's download_batch_s2.py, decisions of 2026-11-10):
  - Collection: COPERNICUS/S2_HARMONIZED (L1C TOA). L2A SR is too sparse over Chile in 2017
    (4-57 scenes per point vs 32-112 for L1C).
  - Dates: the census year (Jan 1 - Dec 31).
  - Cloud mask: COPERNICUS/S2_CLOUD_PROBABILITY < 40, joined on system:index.
  - Terrain (C-) correction with SRTM, coefficients fitted per scene over the batch region (30 m).
  - Per-pixel median of the 10 bands B2-B8A, B11, B12; scaled 0-10000 -> 0-255 uint8;
    clipped to the union of the zone boxes (256 px x 10 m around each zone point).
  - Batches: zone points grouped in 0.5-degree grid cells, so each export is <= ~60 km a side.

    python -m scripts.eo_ml.gee.s2_export batches --year 2017
    python -m scripts.eo_ml.gee.s2_export submit  --year 2017 --batches 0-4     # test
    python -m scripts.eo_ml.gee.s2_export submit  --year 2017                   # all not yet submitted
    python -m scripts.eo_ml.gee.s2_export status  --year 2017

Inputs:  data/B_inter/eo/{year}/zone_points.parquet (scripts/eo_ml/zone_points.py)
Outputs: data/B_inter/eo/{year}/zone_batches.csv (zone_id, lon, lat, batch),
         data/B_inter/eo/{year}/tasks.csv (batch, task_id, description, submitted_at),
         Drive folder eii_s2_{year}: S2_{year}_B{batch:04d}.tif
Run with the lbn-chile environment (earthengine-api).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EO = ROOT / 'data' / 'B_inter' / 'eo'

GEE_PROJECT = 'lbn-chile-496122'
L1C_ID = 'COPERNICUS/S2_HARMONIZED'
CLOUD_ID = 'COPERNICUS/S2_CLOUD_PROBABILITY'
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
CLOUD_PROB_THRESHOLD = 40
SIZE = 256          # pixels per side of each zone box
SCALE = 10          # metres
CELL_DEG = 0.5      # batch grid
M_PER_DEG = 111319.49079327357


# ----------------------------------------------------------------------------- batches

def make_batches(year):
    z = pd.read_parquet(EO / str(year) / 'zone_points.parquet')
    cell = (np.floor(z.lon / CELL_DEG).astype(int).astype(str) + '_' +
            np.floor(z.lat / CELL_DEG).astype(int).astype(str))
    z['batch'] = pd.factorize(cell, sort=True)[0]
    z = z.sort_values(['batch', 'zone_id'])
    z[['zone_id', 'area', 'lon', 'lat', 'batch']].to_csv(EO / str(year) / 'zone_batches.csv', index=False)
    n = z.groupby('batch').size()
    print(f'{len(z):,} zones in {n.size} batches; zones per batch: median {n.median():.0f}, max {n.max()}')


# ----------------------------------------------------------------------------- earth engine

def ee_init():
    import ee
    ee.Initialize(project=GEE_PROJECT)
    return ee


def box(ee, lon, lat):
    h = 0.5 * SIZE * SCALE / M_PER_DEG
    return ee.Geometry.Rectangle([lon - h, lat - h, lon + h, lat + h])


def terrain_correction(ee, region, smooth=5):
    deg2rad = np.pi / 180

    def _tc(img):
        terrain = ee.Terrain.products(ee.Image('USGS/SRTMGL1_003'))
        p = terrain.select('slope').multiply(deg2rad).tan().divide(smooth).atan()
        aspect = terrain.select('aspect').multiply(deg2rad)
        z = ee.Image.constant(ee.Number(img.get('MEAN_SOLAR_ZENITH_ANGLE')).multiply(deg2rad))
        az = ee.Image.constant(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')).multiply(deg2rad))
        cosi = img.expression('cosP*cosZ + sinP*sinZ*cos(az - asp)',
                              {'cosP': p.cos(), 'cosZ': z.cos(), 'sinP': p.sin(), 'sinZ': z.sin(),
                               'az': az, 'asp': aspect}).rename('cosi')
        sub = img.select(BANDS).toFloat()   # same type whether or not the fit succeeds
        fit = ee.Image.cat(ee.Image(1).rename('a'), cosi, sub).reduceRegion(
            reducer=ee.Reducer.linearRegression(numX=2, numY=len(BANDS)),
            geometry=region, scale=30, bestEffort=True, maxPixels=1e9)
        coef = ee.Array(fit.get('coefficients'))
        c = ee.Array(coef.toList().get(0)).divide(ee.Array(coef.toList().get(1)))
        cimg = ee.Image.constant(c.toList()).rename(BANDS)
        corrected = sub.multiply(cimg.add(z.cos())).divide(cimg.add(cosi))
        # scenes with too few valid pixels in the region cannot be fitted: keep them uncorrected
        return ee.Image(ee.Algorithms.If(fit.get('coefficients'), corrected, sub)).copyProperties(img, img.propertyNames())
    return _tc


def batch_image(ee, year, pts):
    region = ee.FeatureCollection([ee.Feature(box(ee, x, y)) for x, y in zip(pts.lon, pts.lat)]).union(1).geometry()
    # census year; if there is no scene at all (offshore islands), widen to year +- 1
    for y0, y1 in [(year, year + 1), (year - 1, year + 2)]:
        flt = ee.Filter.And(ee.Filter.bounds(region), ee.Filter.date(f'{y0}-01-01', f'{y1}-01-01'))
        l1c = (ee.ImageCollection(L1C_ID).filter(flt)
               .filter(ee.Filter.notNull(['MEAN_SOLAR_ZENITH_ANGLE', 'MEAN_SOLAR_AZIMUTH_ANGLE'])))
        if l1c.limit(1).size().getInfo():
            break
    else:
        return None, region, f'no scenes {year - 1}-{year + 1}'
    prob = ee.ImageCollection(CLOUD_ID).filter(flt)
    joined = ee.Join.saveFirst('prob').apply(l1c, prob, ee.Filter.equals(leftField='system:index', rightField='system:index'))

    def mask(img):
        img = ee.Image(img)
        clear = ee.Image(img.get('prob')).select('probability').lt(CLOUD_PROB_THRESHOLD)
        return img.select(BANDS).updateMask(clear).copyProperties(img, img.propertyNames())

    ic = ee.ImageCollection(joined).map(mask).map(terrain_correction(ee, region))
    mosaic = (ic.map(lambda i: i.resample('bilinear')).reduce(ee.Reducer.median())
              .multiply(255.0 / 10000.0).toUint8().rename(BANDS).clip(region))
    return mosaic, region, f'{y0}-{y1 - 1}'


def expand(tokens, all_batches):
    if not tokens:
        return list(all_batches)
    out = set()
    for t in tokens:
        a, _, b = t.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out & set(all_batches))


def submit(year, tokens, folder):
    ee = ee_init()
    d = EO / str(year)
    zb = pd.read_csv(d / 'zone_batches.csv')
    log = d / 'tasks.csv'
    done = set()
    if log.exists():   # submitted batches, except those whose latest task failed or was cancelled
        last = pd.read_csv(log).drop_duplicates('batch', keep='last')
        st = {t['id']: t['state'] for t in ee.data.getTaskList()}
        done = set(last.batch[~last.task_id.map(st).isin(['FAILED', 'CANCELLED'])])
    todo = [b for b in expand(tokens, sorted(zb.batch.unique())) if b not in done]
    print(f'submitting {len(todo)} batches (already submitted: {len(done)})')
    for b in todo:
        pts = zb[zb.batch == b]
        img, region, window = batch_image(ee, year, pts)
        if img is None:
            print(f'  batch {b}: {window}, skipped')
            continue
        desc = f'S2_{year}_B{b:04d}'
        task = ee.batch.Export.image.toDrive(image=img, description=desc, folder=folder, fileNamePrefix=desc,
                                             region=region.bounds(1), scale=SCALE, crs='EPSG:4326',
                                             maxPixels=1e13, fileFormat='GeoTIFF')
        task.start()
        row = pd.DataFrame([dict(batch=b, task_id=task.id, description=desc, n_zones=len(pts), window=window,
                                 submitted_at=time.strftime('%Y-%m-%d %H:%M:%S'))])
        row.to_csv(log, mode='a', header=not log.exists(), index=False)
        print(f'  {desc}: {len(pts)} zones, task {task.id}')


def status(year):
    ee = ee_init()
    log = pd.read_csv(EO / str(year) / 'tasks.csv').drop_duplicates('batch', keep='last')
    st = {t['id']: t for t in ee.data.getTaskList()}
    log['state'] = log.task_id.map(lambda i: st.get(i, {}).get('state', 'UNKNOWN'))
    log['error'] = log.task_id.map(lambda i: st.get(i, {}).get('error_message', ''))
    print(log.state.value_counts().to_string())
    bad = log[log.state.isin(['FAILED', 'CANCELLED'])]
    if len(bad):
        print(bad[['batch', 'description', 'state', 'error']].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['batches', 'submit', 'status'])
    ap.add_argument('--year', type=int, default=2017)
    ap.add_argument('--batches', nargs='*', default=None, help='ids or ranges, e.g. 0-4 7')
    ap.add_argument('--folder', default=None, help='Drive folder (default eii_s2_{year})')
    a = ap.parse_args()
    if a.cmd == 'batches':
        make_batches(a.year)
    elif a.cmd == 'submit':
        submit(a.year, a.batches, a.folder or f'eii_s2_{a.year}')
    else:
        status(a.year)


if __name__ == '__main__':
    main()

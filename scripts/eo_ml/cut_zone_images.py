"""
Cut one SIZE x SIZE image per census zone from the batch mosaics exported by gee/s2_export.py.

    rclone copy gdrive-personal:eii_s2_2017 data/B_inter/eo/2017/s2_batches     # fetch from Drive
    python -m scripts.eo_ml.cut_zone_images --year 2017

Output: data/B_inter/eo/{year}/zones/{zone_id}.tif (10-band uint8, EPSG:4326, deflate) and
        data/B_inter/eo/{year}/zone_images.csv (zone_id, batch, path, empty_share). Zones whose batch
        mosaic is not there yet are skipped; rerunning only cuts what is missing.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PROJ_DATA', str(Path(sys.prefix) / 'share' / 'proj'))

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
EO = ROOT / 'data' / 'B_inter' / 'eo'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=2017)
    ap.add_argument('--size', type=int, default=256)
    a = ap.parse_args()
    d = EO / str(a.year)
    out = d / 'zones'
    out.mkdir(exist_ok=True)
    zb = pd.read_csv(d / 'zone_batches.csv')
    h = a.size // 2
    rows, missing_batches = [], []
    for b, pts in zb.groupby('batch'):
        src = d / 's2_batches' / f'S2_{a.year}_B{b:04d}.tif'
        if not src.exists():
            missing_batches.append(b)
            continue
        with rio.open(src) as r:
            for z in pts.itertuples():
                dst = out / f'{z.zone_id}.tif'
                i, j = r.index(z.lon, z.lat)
                win = Window(j - h, i - h, a.size, a.size)
                arr = r.read(window=win, boundless=True, fill_value=0)
                empty = float((arr.sum(0) == 0).mean())
                if not dst.exists():
                    prof = dict(driver='GTiff', height=a.size, width=a.size, count=r.count, dtype='uint8',
                                crs=r.crs, transform=r.window_transform(win), compress='deflate')
                    with rio.open(dst, 'w', **prof) as w:
                        w.write(arr)
                        w.descriptions = r.descriptions
                rows.append(dict(zone_id=z.zone_id, batch=b, path=str(dst.relative_to(d)), empty_share=empty))
    res = pd.DataFrame(rows)
    res.to_csv(d / 'zone_images.csv', index=False)
    print(f'zones cut: {len(res):,} of {len(zb):,}; batches missing: {len(missing_batches)}')
    if len(res):
        print(f'empty share: mean {res.empty_share.mean():.4f}; zones with >5% empty: {(res.empty_share > .05).sum()}')


if __name__ == '__main__':
    main()

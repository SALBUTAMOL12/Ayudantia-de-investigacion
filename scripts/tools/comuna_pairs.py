#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contiguous pairs of comunas of origin (1992 partition) that differ in their 1992 Mapuche share.

Polygons: BCN comunas (current partition) mapped to the 1992 partition with the crosswalk and dissolved.
Share: population-weighted 1992 Mapuche self-identification share of the closed cells (cohorts 1914-1978)
whose mother lived in the comuna (C_final/groups_panel.parquet). Pairs: queen contiguity, |difference in
share| >= threshold (default 0.1 SD of the cross-comuna distribution, as in the FONDECYT proposal; the
file also records the difference so stricter cuts can be applied downstream).

Output: C_final/comuna_pairs.parquet (pair_id, comuna_a, comuna_b, share_a, share_b, diff, same_region)
Usage: python -m scripts.tools.comuna_pairs
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scripts.directories import data_bcn, data_final
from scripts.tools.comuna_xwalk import to_parent1992


def main(threshold_sd=0.1):
    g = gpd.read_file(os.path.join(data_bcn, 'comunas', 'comunas.shp'))
    g['c92'] = to_parent1992(g.cod_comuna.astype(int)).astype(int)
    g['geometry'] = g.geometry.make_valid()
    g = g.dissolve(by='c92').reset_index()[['c92', 'geometry']]
    g['geometry'] = g.geometry.buffer(0)
    # 1992 Mapuche share of origin comunas (cohorts asked in 1992)
    p = pd.read_parquet(os.path.join(data_final, 'groups_panel.parquet'))
    p = p[(p.year == 1992) & (p.cohort5 >= 1914) & (p.cohort5 <= 1975)]
    sh = p.groupby('mother_comuna').apply(lambda d: pd.Series({'share': np.average(d.mapuche.fillna(0), weights=d.n), 'pop': d.n.sum()}), include_groups=False)
    sh.index = sh.index.astype(int)
    g = g.merge(sh, left_on='c92', right_index=True, how='inner')
    # queen contiguity with a small tolerance (polygons from BCN are topologically clean, but be safe)
    gb = g.copy(); gb['geometry'] = gb.geometry.buffer(50)
    j = gpd.sjoin(gb[['c92', 'geometry']], g[['c92', 'geometry']], predicate='intersects', how='inner')
    j = j[j.c92_left < j.c92_right][['c92_left', 'c92_right']].drop_duplicates()
    j = j.merge(g[['c92', 'share', 'pop']].rename(columns={'c92': 'c92_left', 'share': 'share_a', 'pop': 'pop_a'}), on='c92_left')
    j = j.merge(g[['c92', 'share', 'pop']].rename(columns={'c92': 'c92_right', 'share': 'share_b', 'pop': 'pop_b'}), on='c92_right')
    j['diff'] = (j.share_a - j.share_b).abs()
    sd = g.share.std()
    j['keep_01sd'] = j['diff'] >= threshold_sd * sd
    j['same_region'] = (j.c92_left // 1000) == (j.c92_right // 1000)
    j = j.rename(columns={'c92_left': 'comuna_a', 'c92_right': 'comuna_b'}).reset_index(drop=True)
    j['pair_id'] = np.arange(len(j))
    j.to_parquet(os.path.join(data_final, 'comuna_pairs.parquet'), index=False)
    print(f'comunas with share: {len(g)}; sd of share {sd:.3f}; contiguous pairs {len(j)}; pairs with |diff| >= 0.1 sd ({0.1*sd:.3f}): {int(j.keep_01sd.sum())}; '
          f'>= 5pp: {int((j["diff"] >= 0.05).sum())}; >= 10pp: {int((j["diff"] >= 0.10).sum())}')
    print(j.sort_values('diff', ascending=False).head(10).to_string())
    return j


if __name__ == '__main__':
    main()

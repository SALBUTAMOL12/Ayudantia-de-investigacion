#!/usr/bin/env python3
"""Validate the reconstructed 2002 comuna of residence: comuna-level means of dwelling services and of the
Mapuche share in the reconstructed census vs CASEN 2003 by comuna (313 surveyed comunas). A correct cut gives
correlations near one; a shifted cut gives correlations near the across-region level only."""
import os
import numpy as np
import pandas as pd
from scripts.directories import data_inter, results_tables
from scripts.tools.paper_results import write_macro

HARM = os.path.join(data_inter, 'harmonized')
c = pd.read_parquet(os.path.join(HARM, 'census_2002_hh.parquet'))
s = pd.read_parquet(os.path.join(HARM, 'casen_2003_hh.parquet'))
cp = pd.read_parquet(os.path.join(HARM, 'census_2002_per.parquet'), columns=['comuna', 'age', 'mapuche', 'indigenous'])
sp = pd.read_parquet(os.path.join(HARM, 'casen_2003_per.parquet'), columns=['comuna', 'age', 'mapuche', 'weight'])
vars_ = ['water_public', 'elec_public', 'sanit_sewer', 'floor_dirt', 'wall_solid', 'head_schooling', 'n_people', 'a_phone', 'a_computer']
cm = c.groupby('comuna')[vars_].mean()
sm = s.groupby('comuna').apply(lambda d: pd.Series({v: np.average(d[v].dropna(), weights=d.loc[d[v].notna(), 'weight']) for v in vars_}), include_groups=False)
cm['mapuche'] = cp[cp.age >= 14].groupby('comuna').mapuche.mean()
sm['mapuche'] = sp[sp.age >= 14].groupby('comuna').apply(lambda d: np.average(d.mapuche.dropna(), weights=d.loc[d.mapuche.notna(), 'weight']), include_groups=False)
n = s.groupby('comuna').size()
m = cm.join(sm, lsuffix='_census', rsuffix='_casen', how='inner'); m = m[n.reindex(m.index) >= 50]
rows = []
for v in vars_ + ['mapuche']:
    r = np.corrcoef(m[f'{v}_census'], m[f'{v}_casen'])[0, 1]; rows.append((v, r))
    print(f'{v:16s} corr census-vs-CASEN across {len(m)} comunas: {r:.3f}')
t = ['\\begin{tabular}{lc}', '\\toprule', 'Variable & Correlation across comunas \\\\', '\\midrule'] + [f'{v.replace("_", " ")} & {r:.3f} \\\\' for v, r in rows] + ['\\bottomrule', '\\end{tabular}']
open(os.path.join(results_tables, 'validate_2002_geography.tex'), 'w').write('\n'.join(t) + '\n')
write_macro('geoValComunas', len(m), '{:d}'); write_macro('geoValCorrWater', dict(rows)['water_public'], '{:.2f}'); write_macro('geoValCorrMapuche', dict(rows)['mapuche'], '{:.2f}')

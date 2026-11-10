#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness for the pair DiD using observed survey poverty: CASEN person poverty by comuna of RESIDENCE
(1992, 2017, 2024), adjacent comunas whose 1992 census Mapuche share of residents (persons 14+) differs by
at least 5 points, pair-by-year FE. No imputation involved; the unit is residence rather than origin, and
CASEN 1992 covers 218 comunas, so the sample is smaller.
Output: results/tables/pair_did_casen.tex and macros didCasen*.
"""
import os
import numpy as np
import pandas as pd
import pyreadstat
import geopandas as gpd
from scripts.directories import data_final, data_inter, results_tables, casen_dir
from scripts.tools.comuna_xwalk import to_parent1992
from scripts.analysis.part2_pair_did import pair_did, fmt
from scripts.tools.paper_results import write_macro

HARM = os.path.join(data_inter, 'harmonized')
FIX = {'PAIHUANO': 4105, 'DONIHUE': 6105, 'TREHUACO': 8420, 'ALTO BIO BIO': 8314, 'CUNCO': 9103, 'TEODORO SCHIMDT': 9117,
       'IMPERIAL': 9111, 'COYHAIQUE': 11101, 'PUERTO AISEN': 11201, 'O HIGGINS': 11302, 'CABO DE ORNOS NAVARINO': 12201}


def casen_comuna_poverty():
    rows = []
    xw = pd.read_csv(os.path.join(data_final, 'casen1992_comuna_xwalk.csv'))
    xw['Codigo'] = xw.apply(lambda r: FIX.get(r['name'], r.Codigo), axis=1)
    old2new = dict(zip(xw.comu_old.astype(int), xw.Codigo.astype(int)))
    d, _ = pyreadstat.read_dta(os.path.join(casen_dir, 'casen_1992.dta'), usecols=['comu', 'expr', 'corte'])
    d = d[d.corte.notna()]; d['comuna'] = to_parent1992(d.comu.astype(int).map(old2new)); d['poor'] = d.corte.isin([1, 2]).astype(float); d['w'] = d.expr
    rows.append(d[['comuna', 'poor', 'w']].assign(year=1992))
    for y in [2003, 2017, 2024]:
        h = pd.read_parquet(os.path.join(HARM, f'casen_{y}_per.parquet'), columns=['hh_id', 'comuna92', 'weight'])
        hh = pd.read_parquet(os.path.join(HARM, f'casen_{y}_hh.parquet'), columns=['hh_id', 'poor_official'])
        h = h.merge(hh, on='hh_id'); rows.append(pd.DataFrame({'comuna': h.comuna92, 'poor': h.poor_official, 'w': h.weight, 'year': y}))
    p = pd.concat(rows)
    c = p.groupby(['comuna', 'year']).apply(lambda x: pd.Series({'pov': np.average(x.poor, weights=x.w), 'nobs': len(x)}), include_groups=False).reset_index()
    c['comuna'] = c.comuna.astype(int)
    return c


def residence_share():
    per = pd.read_parquet(os.path.join(HARM, 'census_1992_per.parquet'), columns=['comuna92', 'age', 'mapuche'])
    per = per[(per.age >= 14) & per.mapuche.notna()]
    s = per.groupby('comuna92').mapuche.mean(); s.index = s.index.astype(int)
    return s


def main():
    c = casen_comuna_poverty()
    ok = c.groupby('comuna').year.nunique() == c.year.nunique()
    c = c[c.comuna.isin(ok[ok].index)]
    sh = residence_share()
    pairs = pd.read_parquet(os.path.join(data_final, 'comuna_pairs.parquet'))[['pair_id', 'comuna_a', 'comuna_b']]
    pairs['share_a'] = pairs.comuna_a.map(sh); pairs['share_b'] = pairs.comuna_b.map(sh)
    pairs = pairs.dropna(); pairs = pairs[(pairs.share_a - pairs.share_b).abs() >= 0.05]
    pairs = pairs[pairs.comuna_a.isin(c.comuna) & pairs.comuna_b.isin(c.comuna)]
    rows = []
    for _, p in pairs.iterrows():
        for col, share in [('comuna_a', p.share_a), ('comuna_b', p.share_b)]:
            s = c[c.comuna == int(p[col])].copy(); s['pair'] = p.pair_id; s['share92'] = share; s['pop92'] = s.nobs; rows.append(s)
    d = pd.concat(rows, ignore_index=True)
    r = pair_did(d, 'pov'); rw = pair_did(d, 'pov', weighted=True)
    tex = ['\\begin{tabular}{lcc}', '\\toprule', ' & Unweighted & Weighted by survey observations \\\\', '\\midrule']
    yrs = sorted(d.year.unique())
    for y_ in yrs:
        tex.append(f'$\\beta_{{{y_}}}$ & {fmt(r[f"b{y_}"])} & {fmt(rw[f"b{y_}"])} \\\\')
    tex.append('\\midrule')
    for y_ in yrs[1:]:
        tex.append(f'$\\beta_{{{y_}}}-\\beta_{{1992}}$ & {fmt(r[f"d{y_}"])} & {fmt(rw[f"d{y_}"])} \\\\')
    tex += ['\\midrule', f'Pairs & {r["pairs"]} & \\\\', f'Comunas of residence & {r["comunas"]} & \\\\', '\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, 'pair_did_casen.tex'), 'w').write('\n'.join(tex) + '\n'); print('\n'.join(tex))
    yn = {1992: 'NinetyTwo', 2003: 'TwoThousandThree', 2017: 'Seventeen', 2024: 'TwentyFour'}
    for k, v in r.items():
        if k[0] in 'bd':
            write_macro('didCasen' + ('Beta' if k[0] == 'b' else 'Diff') + yn[int(k[1:])], 100 * v[0]); write_macro('didCasen' + ('Beta' if k[0] == 'b' else 'Diff') + yn[int(k[1:])] + 'Se', 100 * v[1])
    write_macro('didCasenPairs', r['pairs'], '{:d}'); write_macro('didCasenComunas', r['comunas'], '{:d}')


if __name__ == '__main__':
    main()

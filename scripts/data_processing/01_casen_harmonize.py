#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 01: harmonize CASEN microdata (1992, 2003, 2024) into
  B_inter/harmonized/casen_{year}_hh.parquet   one row per household, with the same
  demographic and dwelling block as the census files (02_census_harmonize.py), plus
  income, the official poverty status and the household poverty line.

Income targets (what the official poverty measure of each wave uses):
  1992: ypchaj  = per-capita total household income (CEPAL-adjusted); lines per capita,
        urban 25,750 / rural 17,362 (indigence 12,875 / 9,924). Verified against `corte`.
  2003: ytothaj / numper ; lines urban 43,712 / rural 29,473 (indigence 21,856 / 16,842)
        (Agostini et al. 2010; MIDEPLAN 2005). Verified against `corte` when built.
  2024: ymonecorh = monetary household income; poor if ymonecorh < lp * numper^0.7 with
        lp = 237,094 (non-renters) / 352,359 (renters). Reproduces `pobreza` exactly.

Usage: python -m scripts.data_processing.01_casen_harmonize [--year 1992 2003 2024]
"""
import argparse
import os
import numpy as np
import pandas as pd

from scripts.directories import casen_dir, data_inter, data_final, census_1992
from scripts.tools.hh_features import build_hh_features
from scripts.tools.comuna_xwalk import to_parent1992

OUT = os.path.join(data_inter, 'harmonized')
os.makedirs(OUT, exist_ok=True)


def yesno(s, yes=1):
    s = pd.to_numeric(s, errors='coerce')
    out = (s == yes).astype(float)
    out[s.isna() | (s < 0)] = np.nan
    return out


def isin(s, codes):
    s = pd.to_numeric(s, errors='coerce')
    out = s.isin(codes).astype(float)
    out[s.isna() | (s < 0)] = np.nan
    return out


def region13(code):
    """Region in the 13-region system of 1992 from a post-2007 comuna code."""
    r = (pd.Series(code).astype(float) // 1000)
    return r.replace({15: 1, 14: 10, 16: 8}).to_numpy()


# =============================================================================
def harmonize_2024():
    f = os.path.join(casen_dir, 'casen_2024.dta')
    cols = ['folio', 'id_persona', 'expr', 'estrato', 'region', 'area', 'numper', 'pco1_a', 'sexo', 'edad',
            'esc', 'e1', 'activ', 'o15', 'pueblos_indigenas', 'r3', 'r1a', 'r1b', 'r1b_comuna_esp',
            'v1', 'v2', 'v4', 'v6', 'v27a', 'v20', 'v22', 'v23', 'v23_sistema', 'v24', 'v13', 'v34a', 'v34b',
            'r17a', 'r17b', 'r17c', 'r17d', 'r17e', 'r14', 'ind_hacina',
            'ymonecorh', 'ytotcorh', 'yautcorh', 'ypchtotcor', 'lp', 'li', 'no_arrienda', 'pobreza', 'pobreza_2013']
    print('[2024] loading ...')
    d = pd.read_stata(f, convert_categoricals=False, columns=cols)
    d['hh_id'] = d.folio.astype('int64')
    d['age'] = d.edad.astype(float)
    d['female'] = (d.sexo == 2).astype(float)
    d['head'] = (d.pco1_a == 1).astype(int)
    sch = d.esc.astype(float); sch[(sch < 0) | (d.age < 15)] = np.nan
    d['schooling'] = sch
    emp = yesno(d.activ, 1); emp[d.age < 15] = np.nan
    d['employed'] = emp
    lit = yesno(d.e1, 1); lit[d.age < 15] = np.nan
    d['literate'] = lit
    ind = isin(d.o15, [1, 2]); ind[emp != 1] = np.nan
    d['indep'] = ind
    pi = d.pueblos_indigenas.astype(float)
    d['indigenous'] = np.where(pi == 1, 1.0, np.where(pi == 0, 0.0, np.nan))
    d['mapuche'] = np.where(d.r3 == 4, 1.0, np.where(d.r3.between(1, 12), 0.0, np.nan))
    d['pueblo'] = np.where(d.r3 == 12, 0, d.r3)
    d['foreign_born'] = (d.r1b == 3).astype(float)
    mc = np.where(d.r1b == 1, d.estrato // 100, np.where(d.r1b == 2, d.r1b_comuna_esp, np.nan)).astype(float)
    mc[mc < 1000] = np.nan
    d['mother_comuna'] = to_parent1992(mc)
    d['comuna'] = (d.estrato // 100).astype(float)
    d['comuna92'] = to_parent1992(d.comuna)
    d['region13'] = region13(d.comuna)
    d['urban'] = (d.area == 1).astype(float)
    d['weight'] = d.expr.astype(float)

    hh = build_hh_features(d[['hh_id', 'head', 'female', 'age', 'schooling', 'employed', 'literate', 'indep', 'indigenous']])
    h = d.drop_duplicates('hh_id').set_index('hh_id')
    H = pd.DataFrame(index=h.index)
    H['region'] = h.region.astype(float); H['region13'] = h.region13; H['comuna'] = h.comuna; H['comuna92'] = h.comuna92
    H['urban'] = h.urban; H['weight'] = h.weight
    H['dw_house'] = isin(h.v1, [1, 2, 3]); H['dw_apt'] = isin(h.v1, [4, 5]); H['dw_precarious'] = isin(h.v1, [6, 7, 8, 9, 10])
    H['wall_solid'] = isin(h.v2, [1, 2]); H['wall_wood'] = isin(h.v2, [3]); H['wall_precarious'] = isin(h.v2, [4, 5, 6])
    H['roof_good'] = isin(h.v6, [1, 2]); H['roof_zinc'] = isin(h.v6, [3, 4]); H['roof_precarious'] = isin(h.v6, [5, 6, 7, 8])
    H['floor_good'] = isin(h.v4, [1, 2, 3]); H['floor_cement'] = isin(h.v4, [4, 5]); H['floor_dirt'] = isin(h.v4, [6])
    H['water_public'] = yesno(h.v20, 1); H['water_inside'] = yesno(h.v22, 1)
    H['sanit_sewer'] = ((h.v23 == 1) & (h.v23_sistema == 1)).astype(float); H['sanit_none'] = yesno(h.v23, 2)
    H['elec_public'] = isin(h.v24, [1, 2, 3, 4, 5]); H['elec_none'] = yesno(h.v24, 8)
    nb = h.v27a.astype(float); nb[nb < 0] = np.nan; H['n_bedrooms'] = nb.clip(upper=6)
    H['hacinamiento'] = isin(h.ind_hacina, [2, 3])
    H['tenure_owned'] = yesno(h.v13, 1); H['tenure_rented'] = yesno(h.v13, 2)
    H['fuel_wood'] = yesno(h.v34a, 4); H['heat_wood'] = yesno(h.v34b, 4)
    H['a_internet_fixed'] = yesno(h.r17a, 1)
    H['a_internet_mobile'] = ((h.r17b == 1) | (h.r17c == 1)).astype(float)
    H['a_internet_sat'] = yesno(h.r17e, 1)
    H['a_tablet'] = yesno(h.r17d, 1)
    H['a_vehicle'] = yesno(h.r14, 1)
    H['numper'] = h.numper.astype(float)
    H['y_hh'] = h.ymonecorh.astype(float)          # official target income
    H['y_hh_total'] = h.ytotcorh.astype(float)
    H['y_hh_aut'] = h.yautcorh.astype(float)
    H['ypc'] = H.y_hh / H.numper
    H['renter'] = (h.no_arrienda == 0).astype(float)
    H['lp_pc'] = h.lp.astype(float); H['li_pc'] = h.li.astype(float)
    H['line_hh'] = H.lp_pc * H.numper ** 0.7
    H['line_ext_hh'] = H.li_pc * H.numper ** 0.7
    H['poor_official'] = h.pobreza.isin([1, 2]).astype(float)
    H['extreme_official'] = (h.pobreza == 1).astype(float)
    H['poor_calc'] = (H.y_hh < H.line_hh).astype(float)
    H['extreme_calc'] = (H.y_hh < H.line_ext_hh).astype(float)
    H['poor_2013'] = h.pobreza_2013.isin([1, 2]).astype(float)
    hh = hh.merge(H.reset_index(), on='hh_id', how='left')
    hh['year'] = 2024
    print('[2024] poor_calc==poor_official:', (hh.poor_calc == hh.poor_official).mean(),
          ' extreme:', (hh.extreme_calc == hh.extreme_official).mean())
    hh.to_parquet(os.path.join(OUT, 'casen_2024_hh.parquet'), index=False)
    d[['hh_id', 'id_persona', 'weight', 'region', 'comuna', 'comuna92', 'urban', 'female', 'age', 'head', 'schooling',
       'employed', 'indigenous', 'mapuche', 'pueblo', 'foreign_born', 'mother_comuna']].to_parquet(
        os.path.join(OUT, 'casen_2024_per.parquet'), index=False)
    print('[2024] hh', len(hh)); print(hh.describe().T.to_string())


# =============================================================================
def harmonize_1992():
    import pyreadstat
    f = os.path.join(casen_dir, 'casen_1992.dta')
    cols = ['r', 'comu', 'z', 'f', 'o', 'expr', 'numper', 'pco1', 'sexo', 'edad', 'esc', 'e1', 'o21', 'o7',
            'v4', 'v5', 'v9', 'v24', 'v11', 'v12', 'v13', 'v14', 'v15', 'v30', 'v32', 'v34', 'corte', 'ypchaj', 'yauthaj', 'ymonehaj']
    print('[1992] loading ...')
    d, meta = pyreadstat.read_dta(f, usecols=cols)
    d['hh_id'] = d.f.astype('int64')
    d['age'] = d.edad.astype(float)
    d['female'] = (d.sexo == 2).astype(float)
    d['head'] = (d.pco1 == 1).astype(int)
    sch = d.esc.astype(float); sch[(sch < 0) | (d.age < 15)] = np.nan
    d['schooling'] = sch
    emp = yesno(d.o21, 1); emp[d.age < 15] = np.nan
    d['employed'] = emp
    lit = yesno(d.e1, 1); lit[d.age < 15] = np.nan
    d['literate'] = lit
    ind = isin(d.o7, [1, 2]); ind[emp != 1] = np.nan
    d['indep'] = ind
    d['indigenous'] = np.nan      # not asked in 1992
    d['urban'] = (d.z == 1).astype(float)
    d['weight'] = d.expr.astype(float)
    hh = build_hh_features(d[['hh_id', 'head', 'female', 'age', 'schooling', 'employed', 'literate', 'indep', 'indigenous']])
    h = d.drop_duplicates('hh_id').set_index('hh_id')
    H = pd.DataFrame(index=h.index)
    H['region13'] = h.r.astype(float); H['region'] = H.region13; H['comuna_old'] = h.comu.astype(float)
    H['urban'] = h.urban; H['weight'] = h.weight
    H['dw_house'] = yesno(h.v14, 1); H['dw_apt'] = yesno(h.v14, 2); H['dw_precarious'] = isin(h.v14, [3, 4, 5, 6, 7])
    H['wall_solid'] = isin(h.v30, [1, 2, 5]); H['wall_wood'] = isin(h.v30, [3, 7, 8]); H['wall_precarious'] = isin(h.v30, [0, 4, 6, 9])
    H['roof_good'] = isin(h.v34, [1, 3]); H['roof_zinc'] = isin(h.v34, [2, 4, 7]); H['roof_precarious'] = isin(h.v34, [5, 6, 8, 9])
    for c in ['roof_good', 'roof_zinc', 'roof_precarious']:
        H.loc[h.v34 == 0, c] = np.nan
    H['floor_good'] = isin(h.v32, [1, 3, 4]); H['floor_cement'] = isin(h.v32, [2, 6]); H['floor_dirt'] = isin(h.v32, [5, 7, 8])
    for c in ['floor_good', 'floor_cement', 'floor_dirt']:
        H.loc[h.v32 == 9, c] = np.nan
    H['water_public'] = isin(h.v12, [1, 2, 3]); H['water_inside'] = isin(h.v12, [1, 4])
    for c in ['water_public', 'water_inside']:
        H.loc[h.v12 == 7, c] = np.nan
    H['sanit_sewer'] = yesno(h.v13, 1); H['sanit_none'] = yesno(h.v13, 5)
    H.loc[h.v13 == 6, ['sanit_sewer', 'sanit_none']] = np.nan
    H['elec_public'] = isin(h.v11, [1, 2, 3]); H['elec_none'] = yesno(h.v11, 4)
    H.loc[h.v11 == 5, ['elec_public', 'elec_none']] = np.nan
    H['n_bedrooms'] = h.v5.astype(float).clip(upper=6)
    H['n_rooms'] = (h.v4.astype(float) - h.v9.astype(float).fillna(0)).clip(lower=1, upper=10)   # census counts rooms excluding bathrooms
    H['tenure_owned'] = isin(h.v15, [1, 2]); H['tenure_rented'] = isin(h.v15, [3, 4])
    H['numper'] = h.numper.astype(float)
    H['ypc'] = h.ypchaj.astype(float)
    H['y_hh'] = H.ypc * H.numper
    H['y_hh_aut'] = h.yauthaj.astype(float); H['y_hh_mon'] = h.ymonehaj.astype(float)
    H['lp_pc'] = np.where(H.urban == 1, 25750.0, 17362.0)
    H['li_pc'] = np.where(H.urban == 1, 12875.0, 9924.0)
    H['line_hh'] = H.lp_pc * H.numper; H['line_ext_hh'] = H.li_pc * H.numper
    H['poor_official'] = np.where(h.corte.isna(), np.nan, h.corte.isin([1, 2]).astype(float))
    H['extreme_official'] = np.where(h.corte.isna(), np.nan, (h.corte == 1).astype(float))
    H['poor_calc'] = (H.ypc < H.lp_pc).astype(float); H['extreme_calc'] = (H.ypc < H.li_pc).astype(float)
    hh = hh.merge(H.reset_index(), on='hh_id', how='left')
    hh['year'] = 1992
    ok = hh.poor_official.notna()
    print('[1992] poor_calc==poor_official:', (hh.poor_calc[ok] == hh.poor_official[ok]).mean(),
          ' extreme:', (hh.extreme_calc[ok] == hh.extreme_official[ok]).mean())
    hh.to_parquet(os.path.join(OUT, 'casen_1992_hh.parquet'), index=False)
    print('[1992] hh', len(hh)); print(hh.describe().T.to_string())


def harmonize_2017():
    f = os.path.join(casen_dir, 'casen_2017.dta')
    cols = ['folio', 'o', 'expr', 'estrato', 'region', 'area', 'numper', 'pco1', 'sexo', 'edad', 'esc', 'e1', 'activ', 'o15',
            'r3', 'r1b', 'r1b_c_cod', 'v1', 'v2', 'v4', 'v6', 'v27a', 'v20', 'v22', 'v24', 'v13',
            'ymonecorh', 'ytotcorh', 'ypch', 'lp', 'li', 'no_arrienda', 'pobreza']
    print('[2017] loading ...')
    d = pd.read_stata(f, convert_categoricals=False, columns=cols)
    d['hh_id'] = d.folio.astype('int64')
    d['age'] = d.edad.astype(float)
    d['female'] = (d.sexo == 2).astype(float)
    d['head'] = (d.pco1 == 1).astype(int)
    sch = d.esc.astype(float); sch[(sch < 0) | (sch > 30) | (d.age < 15)] = np.nan
    d['schooling'] = sch
    emp = yesno(d.activ, 1); emp[d.age < 15] = np.nan
    d['employed'] = emp
    lit = yesno(d.e1, 1); lit[(d.age < 15) | (d.e1 == 9)] = np.nan
    d['literate'] = lit
    ind = isin(d.o15, [1, 2]); ind[emp != 1] = np.nan
    d['indep'] = ind
    r3 = d.r3.astype(float)
    d['indigenous'] = np.where(r3.between(1, 9), 1.0, np.where(r3 == 10, 0.0, np.nan))
    d['mapuche'] = np.where(r3 == 4, 1.0, np.where(r3.between(1, 10), 0.0, np.nan))
    R3 = {1: 2, 2: 3, 3: 5, 4: 1, 5: 4, 6: 6, 7: 8, 8: 9, 9: 7}
    d['pueblo'] = np.where(r3 == 10, 0, r3.map(R3).fillna(-99))
    d['foreign_born'] = (d.r1b == 3).astype(float)
    d['comuna'] = (d.estrato // 10).astype(float)
    mc = np.where(d.r1b == 1, d.comuna, np.where(d.r1b == 2, d.r1b_c_cod, np.nan)).astype(float)
    mc[mc < 1000] = np.nan
    d['mother_comuna'] = to_parent1992(mc)
    d['comuna92'] = to_parent1992(d.comuna)
    d['region13'] = region13(d.comuna)
    d['urban'] = (d.area == 1).astype(float)
    d['weight'] = d.expr.astype(float)
    hh = build_hh_features(d[['hh_id', 'head', 'female', 'age', 'schooling', 'employed', 'literate', 'indep', 'indigenous']])
    h = d.drop_duplicates('hh_id').set_index('hh_id')
    H = pd.DataFrame(index=h.index)
    H['region'] = h.region.astype(float); H['region13'] = h.region13; H['comuna'] = h.comuna; H['comuna92'] = h.comuna92
    H['urban'] = h.urban; H['weight'] = h.weight
    H['dw_house'] = isin(h.v1, [1, 2, 3]); H['dw_apt'] = isin(h.v1, [4, 5]); H['dw_precarious'] = isin(h.v1, [6, 7, 8, 9, 10])
    H['wall_solid'] = isin(h.v2, [1, 2]); H['wall_wood'] = isin(h.v2, [3]); H['wall_precarious'] = isin(h.v2, [4, 5, 6])
    H['roof_good'] = isin(h.v6, [1, 2]); H['roof_zinc'] = isin(h.v6, [3, 4]); H['roof_precarious'] = isin(h.v6, [5, 6, 7, 8])
    H['floor_good'] = isin(h.v4, [1, 2, 3]); H['floor_cement'] = isin(h.v4, [4, 5]); H['floor_dirt'] = isin(h.v4, [6])
    for c, v in [('roof_good', 'v6'), ('roof_zinc', 'v6'), ('roof_precarious', 'v6'), ('floor_good', 'v4'), ('floor_cement', 'v4'), ('floor_dirt', 'v4')]:
        H.loc[h[v] == 9, c] = np.nan
    H['water_public'] = isin(h.v20, [1, 2, 3]); H['water_inside'] = yesno(h.v22, 1)
    H.loc[h.v20 == 9, 'water_public'] = np.nan; H.loc[h.v22 == 9, 'water_inside'] = np.nan
    H['elec_public'] = isin(h.v24, [1, 2, 3, 4])
    nb = h.v27a.astype(float); nb[(nb < 0) | (nb == 99)] = np.nan; H['n_bedrooms'] = nb.clip(upper=6)
    H['tenure_owned'] = isin(h.v13, [1, 2, 3, 4]); H['tenure_rented'] = isin(h.v13, [5, 6])
    H['numper'] = h.numper.astype(float)
    H['y_hh'] = h.ymonecorh.astype(float); H['y_hh_total'] = h.ytotcorh.astype(float)
    H['ypc'] = H.y_hh / H.numper
    H['renter'] = (h.no_arrienda == 0).astype(float)
    H['lp_pc'] = h.lp.astype(float); H['li_pc'] = h.li.astype(float)
    H['line_hh'] = H.lp_pc * H.numper ** 0.7; H['line_ext_hh'] = H.li_pc * H.numper ** 0.7
    H['poor_official'] = h.pobreza.isin([1, 2]).astype(float); H['extreme_official'] = (h.pobreza == 1).astype(float)
    H['poor_calc'] = (H.y_hh < H.line_hh).astype(float); H['extreme_calc'] = (H.y_hh < H.line_ext_hh).astype(float)
    hh = hh.merge(H.reset_index(), on='hh_id', how='left')
    hh['year'] = 2017
    print('[2017] poor_calc==poor_official:', (hh.poor_calc == hh.poor_official).mean(), ' extreme:', (hh.extreme_calc == hh.extreme_official).mean())
    print('[2017] lines by renter status:', hh.groupby('renter')[['lp_pc', 'li_pc']].first().to_dict())
    hh.to_parquet(os.path.join(OUT, 'casen_2017_hh.parquet'), index=False)
    d[['hh_id', 'o', 'weight', 'region', 'comuna', 'comuna92', 'urban', 'female', 'age', 'head', 'schooling', 'employed',
       'indigenous', 'mapuche', 'pueblo', 'foreign_born', 'mother_comuna']].rename(columns={'o': 'id_persona'}).to_parquet(
        os.path.join(OUT, 'casen_2017_per.parquet'), index=False)
    print('[2017] hh', len(hh)); print(hh.describe().T.to_string())


def harmonize_2003():
    f = os.path.join(casen_dir, 'casen_2003.dta')
    cols = ['segmento', 'f', 'o', 'comu', 'z', 'expr', 'numper', 'pco1', 'sexo', 'edad', 'esc', 'e1', 'activ', 'o9', 'r25',
            'v3a', 'v3b', 'v3c', 'v3d', 'v3e', 'v3f', 'v3h', 'v4', 'v5', 'v6', 'v7', 'v8a', 'v9a', 'v10a', 'v11', 'v12', 'r10a', 'r10b', 'r10c', 'r10d', 'r10e', 'r10f',
            'r10g', 'r10h', 'r10i', 'r10j', 'r10k', 'r11', 'ytothaj', 'corte']
    print('[2003] loading ...')
    d = pd.read_stata(f, convert_categoricals=False, columns=cols)
    d['hh_id'] = pd.factorize(pd.MultiIndex.from_frame(d[['segmento', 'f']]))[0].astype('int64')
    d['age'] = d.edad.astype(float)
    d['female'] = (d.sexo == 2).astype(float)
    d['head'] = (d.pco1 == 1).astype(int)
    sch = d.esc.astype(float); sch[(sch < 0) | (sch > 30) | (d.age < 15)] = np.nan
    d['schooling'] = sch
    emp = yesno(d.activ, 1); emp[d.age < 15] = np.nan
    d['employed'] = emp
    lit = yesno(d.e1, 1); lit[(d.age < 15) | (d.e1 == 9)] = np.nan
    d['literate'] = lit
    ind = isin(d.o9, [1, 2]); ind[emp != 1] = np.nan
    d['indep'] = ind
    r = d.r25.astype(float)
    d['indigenous'] = np.where(r.between(1, 8), 1.0, np.where(r == 0, 0.0, np.nan))
    d['mapuche'] = np.where(r == 4, 1.0, np.where(r.between(0, 8), 0.0, np.nan))
    R25 = {1: 2, 2: 3, 3: 5, 4: 1, 5: 4, 6: 6, 7: 8, 8: 9}
    d['pueblo'] = np.where(r == 0, 0, r.map(R25).fillna(-99))
    d['foreign_born'] = np.nan; d['mother_comuna'] = np.nan
    # 2002-vintage ('old') comuna codes -> post-2007 codes by name (same crosswalk logic as the 1992 wave)
    import re, unicodedata
    def norm(x): return re.sub(r'[^A-Z ]', '', unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().upper()).strip()
    lab = pd.read_csv(os.path.join(data_final, 'casen2003_comuna_labels.csv'))
    FIX = {'PAIHUANO': 'PAIGUANO', 'DONIHUE': 'DONIGUE', 'TREHUACO': 'TREGUACO', 'CUNCO': 'CIUNCO', 'TEODORO SCHIMDT': 'TEODORO SCHMIDT',
           'IMPERIAL': 'NUEVA IMPERIAL', 'COYHAIQUE': 'COIHAIQUE', 'PUERTO AISEN': 'AISEN', 'O HIGGINS': 'OHIGGINS', 'CABO DE ORNOS NAVARINO': 'CABO DE HORNO',
           'ALTO BIO BIO': 'ALTO BIOBIO', 'AYSEN': 'AISEN'}
    lab['name_m'] = lab.name.map(lambda x: FIX.get(x, x))
    new92 = pd.read_csv(os.path.join(census_1992, 'microdata', 'COMUNAS.csv'), index_col=0)
    new92['name_m'] = new92.Nombre.map(norm)
    xw = dict(zip(lab.old, lab.merge(new92[['Codigo', 'name_m']], on='name_m', how='left').Codigo))
    d['comuna_old'] = d.comu.astype(float)
    d['comuna'] = d.comu.map(xw).astype(float)
    assert d.comuna.notna().all(), d.loc[d.comuna.isna(), 'comu'].unique()
    d['comuna92'] = to_parent1992(d.comuna)
    d['region13'] = (d.comu // 1000).astype(float)
    d['urban'] = (d.z == 1).astype(float)
    d['weight'] = d.expr.astype(float)
    hh = build_hh_features(d[['hh_id', 'head', 'female', 'age', 'schooling', 'employed', 'literate', 'indep', 'indigenous']])
    h = d.drop_duplicates('hh_id').set_index('hh_id')
    H = pd.DataFrame(index=h.index)
    H['region13'] = h.region13; H['region'] = h.region13; H['comuna'] = h.comuna; H['comuna92'] = h.comuna92; H['comuna_old'] = h.comuna_old
    H['urban'] = h.urban; H['weight'] = h.weight
    H['dw_house'] = isin(h.v11, [1, 2]); H['dw_apt'] = yesno(h.v11, 3); H['dw_precarious'] = isin(h.v11, [4, 5, 6, 7, 8, 9])
    H['wall_solid'] = isin(h.v8a, [1, 2]); H['wall_wood'] = isin(h.v8a, [3, 5]); H['wall_precarious'] = isin(h.v8a, [4, 6, 7, 8])
    H['roof_good'] = yesno(h.v10a, 1); H['roof_zinc'] = isin(h.v10a, [2, 3]); H['roof_precarious'] = isin(h.v10a, [4, 5, 6])
    H['floor_good'] = isin(h.v9a, [1, 3]); H['floor_cement'] = isin(h.v9a, [2, 4]); H['floor_dirt'] = yesno(h.v9a, 5)
    for c, v in [('wall_solid', 'v8a'), ('wall_wood', 'v8a'), ('wall_precarious', 'v8a'), ('roof_good', 'v10a'), ('roof_zinc', 'v10a'), ('roof_precarious', 'v10a'),
                 ('floor_good', 'v9a'), ('floor_cement', 'v9a'), ('floor_dirt', 'v9a')]:
        H.loc[h[v] == 9, c] = np.nan
    H['water_public'] = isin(h.v4, [1, 2, 3]); H['water_inside'] = yesno(h.v5, 1)
    H['sanit_sewer'] = isin(h.v6, [1, 2]); H['sanit_none'] = yesno(h.v6, 7)
    H['elec_public'] = isin(h.v7, [1, 2, 3]); H['elec_none'] = yesno(h.v7, 7)
    for c, v in [('water_public', 'v4'), ('water_inside', 'v5'), ('sanit_sewer', 'v6'), ('sanit_none', 'v6'), ('elec_public', 'v7'), ('elec_none', 'v7')]:
        H.loc[h[v] == 9, c] = np.nan
    nb = h.v3a.astype(float); nb[nb > 20] = np.nan; H['n_bedrooms'] = nb.clip(upper=6)
    nr = h[['v3a', 'v3b', 'v3c', 'v3d', 'v3e', 'v3f', 'v3h']].astype(float).clip(upper=20).sum(axis=1)   # all rooms except bathrooms (census 2002 definition)
    H['n_rooms'] = nr.clip(lower=1, upper=10)
    H['tenure_owned'] = isin(h.v12, [1, 2, 3, 4]); H['tenure_rented'] = isin(h.v12, [5, 6])
    H.loc[h.v12 == 99, ['tenure_owned', 'tenure_rented']] = np.nan
    for a, v in [('a_washer', 'r10a'), ('a_fridge', 'r10b'), ('a_phone', 'r10c'), ('a_video', 'r10d'), ('a_microwave', 'r10e'), ('a_computer', 'r10f'),
                 ('a_calefont', 'r10i')]:
        H[a] = yesno(h[v], 1)
    H['a_internet'] = ((h.r10g == 1) | (h.r10h == 1)).astype(float)
    H['a_cable_tv'] = ((h.r10j == 1) | (h.r10k == 1)).astype(float)
    H['a_cellphone'] = isin(h.r11, [1, 2]); H.loc[h.r11 == 9, 'a_cellphone'] = np.nan
    H['numper'] = h.numper.astype(float)
    H['y_hh'] = h.ytothaj.astype(float); H['ypc'] = H.y_hh / H.numper
    H['lp_pc'] = np.where(H.urban == 1, 43712.0, 29473.0); H['li_pc'] = np.where(H.urban == 1, 21856.0, 16842.0)
    H['line_hh'] = H.lp_pc * H.numper; H['line_ext_hh'] = H.li_pc * H.numper
    H['poor_official'] = np.where(h.corte.isna(), np.nan, h.corte.isin([1, 2]).astype(float))
    H['extreme_official'] = np.where(h.corte.isna(), np.nan, (h.corte == 1).astype(float))
    H['poor_calc'] = (H.ypc < H.lp_pc).astype(float); H['extreme_calc'] = (H.ypc < H.li_pc).astype(float)
    hh = hh.merge(H.reset_index(), on='hh_id', how='left'); hh['year'] = 2003
    ok = hh.poor_official.notna()
    print('[2003] poor_calc==poor_official:', (hh.poor_calc[ok] == hh.poor_official[ok]).mean(), ' extreme:', (hh.extreme_calc[ok] == hh.extreme_official[ok]).mean())
    hh.to_parquet(os.path.join(OUT, 'casen_2003_hh.parquet'), index=False)
    d[['hh_id', 'o', 'weight', 'region13', 'comuna', 'comuna92', 'urban', 'female', 'age', 'head', 'schooling', 'employed', 'indigenous', 'mapuche',
       'pueblo', 'foreign_born', 'mother_comuna']].rename(columns={'o': 'id_persona', 'region13': 'region'}).to_parquet(os.path.join(OUT, 'casen_2003_per.parquet'), index=False)
    print('[2003] hh', len(hh)); print(hh.describe().T.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, nargs='*', default=[1992, 2017, 2024])
    a = ap.parse_args()
    for y in a.year:
        {1992: harmonize_1992, 2003: harmonize_2003, 2017: harmonize_2017, 2024: harmonize_2024}[y]()

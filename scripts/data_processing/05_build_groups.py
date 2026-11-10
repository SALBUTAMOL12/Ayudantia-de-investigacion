#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 05: closed cohort groups g = (5-year birth cohort, sex, comuna where the mother
lived at birth; 1992 comuna vintage, post-1992 splits merged into parents).

From each census person file build, per group: population, indigenous self-identification
share (and Mapuche share), mean schooling of adults, share urban residents, share living in
the mother's comuna; then merge the simulated poverty (script 04, spec B and A).

Output: C_final/groups_panel.parquet (one row per group x census year)
        C_final/groups_wide.parquet  (one row per group, 1992 and 2024 columns side by side)

Sample for the identification comparison: cohorts born 1914-1978 (aged 14+ in 1992, when the
ethnicity question was only asked to people aged 14 and over) and alive/observed in both
censuses; mother's comuna in Chile.
"""
import os
import numpy as np
import pandas as pd

from scripts.directories import data_inter, data_final
from scripts.tools.comuna_xwalk import to_parent1992

HARM = os.path.join(data_inter, 'harmonized')
SIM = os.path.join(data_inter, 'simulations')
KEY = ['cohort5', 'female', 'mother_comuna']


def group_stats(year):
    per = pd.read_parquet(os.path.join(HARM, f'census_{year}_per.parquet'))
    per['mother_comuna'] = to_parent1992(per.mother_comuna); per['comuna92'] = to_parent1992(per.comuna92)   # idempotent; picks up crosswalk updates
    # household-based measure: lives with at least one indigenous adult (defined for children too)
    ad = per[(per.age >= 14) & per.indigenous.notna()].groupby('hh_id').agg(indig_hh=('indigenous', 'max'), mapuche_hh=('mapuche', 'max'))
    per = per.merge(ad, on='hh_id', how='left')
    # directly observed dwelling deprivation of the person's household (no imputation involved)
    hhf = pd.read_parquet(os.path.join(HARM, f'census_{year}_hh.parquet'))
    dep_cols = [c for c in ['floor_dirt', 'sanit_none', 'elec_none', 'water_public', 'wall_precarious', 'roof_precarious', 'n_bedrooms', 'n_people'] if c in hhf]
    per = per.merge(hhf[['hh_id'] + dep_cols], on='hh_id', how='left')
    if 'n_bedrooms' in per and 'n_people' in per:
        per['persons_per_bedroom'] = per.n_people / per.n_bedrooms.clip(lower=1)
    per = per[per[KEY].notna().all(axis=1)].copy()
    per['adult'] = (per.age >= 15).astype(float)
    per['sch'] = per.schooling
    per['hs'] = np.where(per.schooling.notna(), (per.schooling >= 12).astype(float), np.nan)
    dep = [c for c in ['floor_dirt', 'sanit_none', 'elec_none'] if c in per]
    per['deprived'] = per[dep].max(axis=1) if dep else np.nan
    per['in_mother_comuna'] = (per.comuna92 == per.mother_comuna).astype(float)
    per['eth_asked'] = per.indigenous.notna().astype(float)
    g = per.groupby(KEY).agg(n=('age', 'size'), n_eth_asked=('eth_asked', 'sum'),
                             indigenous=('indigenous', 'mean'), mapuche=('mapuche', 'mean'), indig_hh=('indig_hh', 'mean'), mapuche_hh=('mapuche_hh', 'mean'),
                             schooling=('sch', 'mean'), urban=('urban', 'mean'),
                             in_mother_comuna=('in_mother_comuna', 'mean'),
                             employed=('employed', 'mean'), age=('age', 'mean'), hs=('hs', 'mean'), deprived=('deprived', 'mean'),
                             **{c: (c, 'mean') for c in ['floor_dirt', 'sanit_none', 'elec_none', 'water_public', 'wall_precarious', 'roof_precarious', 'persons_per_bedroom'] if c in per}).reset_index()
    g['year'] = year
    for spec in ['A', 'B']:
        f = os.path.join(SIM, f'{year}_{spec}_groups.parquet')
        if os.path.exists(f):
            s = pd.read_parquet(f)[KEY + ['poor', 'poor_sd', 'fgt1', 'fgt1_sd']].rename(
                columns={'poor': f'poor_{spec}', 'poor_sd': f'poor_sd_{spec}', 'fgt1': f'fgt1_{spec}', 'fgt1_sd': f'fgt1_sd_{spec}'})
            g = g.merge(s, on=KEY, how='left')
    return g


if __name__ == '__main__':
    parts = [group_stats(y) for y in [1992, 2017, 2024] if os.path.exists(os.path.join(HARM, f'census_{y}_per.parquet'))]
    panel = pd.concat(parts, ignore_index=True)
    panel.to_parquet(os.path.join(data_final, 'groups_panel.parquet'), index=False)
    wide = None
    for y in sorted(panel.year.unique()):
        s = panel[panel.year == y].drop(columns='year').set_index(KEY).add_suffix(f'_{y}')
        wide = s if wide is None else wide.join(s, how='outer')
    wide = wide.reset_index()
    wide.to_parquet(os.path.join(data_final, 'groups_wide.parquet'), index=False)
    print('groups:', panel.groupby('year').size().to_dict())
    both = wide.dropna(subset=['n_1992', 'n_2024'])
    both = both[(both.cohort5 >= 1914) & (both.cohort5 <= 1975)]
    print('groups observed in both censuses, cohorts 1914-1978:', len(both), 'persons 1992:', both.n_1992.sum(), '2024:', both.n_2024.sum())
    w = both.n_1992
    print('indigenous share 1992 (pop-weighted):', np.average(both.indigenous_1992.fillna(0), weights=w),
          ' 2024:', np.average(both.indigenous_2024.fillna(0), weights=both.n_2024))
    print(both[['n_1992', 'n_2024', 'indigenous_1992', 'indigenous_2024', 'mapuche_1992', 'mapuche_2024']].describe().T.to_string())

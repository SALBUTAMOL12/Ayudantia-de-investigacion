#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Household demographic features from a standardized person table.

Input: a DataFrame with one row per person and these columns
    hh_id      : household key
    head       : 1 if household head
    female     : 1/0
    age        : years (NaN allowed)
    schooling  : years of schooling for persons aged 15+, NaN otherwise/missing
    employed   : 1/0 for persons 15+, NaN otherwise/missing
    literate   : 1/0 (NaN if not asked)
    indep      : 1/0 employer or self-employed (NaN if not employed/not asked)
    indigenous : 1/0 self-identified (NaN if not asked)

Output: one row per hh_id with the harmonized demographic block used by every
wave (census and CASEN), so first-stage models are specified identically.
"""
import numpy as np
import pandas as pd


def build_hh_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p['kid'] = (p['age'] <= 14).astype(float)
    p['old'] = (p['age'] >= 65).astype(float)
    p['adult'] = (p['age'] >= 15).astype(float)
    p['one'] = 1.0
    g = p.groupby('hh_id', sort=False)
    out = g.agg(n_people=('one', 'sum'),
                n_kids=('kid', 'sum'),
                n_old=('old', 'sum'),
                n_adults=('adult', 'sum'),
                mean_age=('age', 'mean'),
                mean_schooling_adults=('schooling', 'mean'),
                max_schooling=('schooling', 'max'),
                n_employed=('employed', 'sum'),
                frac_literate_adults=('literate', 'mean'),
                n_indigenous=('indigenous', 'sum'))
    out['frac_kids'] = out['n_kids'] / out['n_people']
    out['frac_old'] = out['n_old'] / out['n_people']
    out['frac_employed_adults'] = np.where(out['n_adults'] > 0, out['n_employed'] / out['n_adults'].clip(lower=1), np.nan)
    out['dep_ratio'] = (out['n_kids'] + out['n_old']) / (out['n_people'] - out['n_kids'] - out['n_old']).clip(lower=1)
    out['frac_indigenous'] = out['n_indigenous'] / out['n_people']
    # head characteristics (first head row per household; if none, oldest adult)
    heads = p[p['head'] == 1].drop_duplicates('hh_id').set_index('hh_id')
    out['head_female'] = heads['female'].reindex(out.index)
    out['head_age'] = heads['age'].reindex(out.index)
    out['head_schooling'] = heads['schooling'].reindex(out.index)
    out['head_employed'] = heads['employed'].reindex(out.index)
    out['head_literate'] = heads['literate'].reindex(out.index)
    out['head_indep'] = heads['indep'].reindex(out.index).fillna(0)
    out['head_indigenous'] = heads['indigenous'].reindex(out.index)
    out['n_people_sq'] = out['n_people'] ** 2
    return out.reset_index()


DEMOG_VARS = ['n_people', 'n_people_sq', 'n_kids', 'frac_kids', 'n_old', 'frac_old', 'mean_age',
              'head_female', 'head_age', 'head_schooling', 'mean_schooling_adults', 'max_schooling',
              'head_employed', 'n_employed', 'frac_employed_adults', 'dep_ratio',
              'head_literate', 'frac_literate_adults', 'head_indep']

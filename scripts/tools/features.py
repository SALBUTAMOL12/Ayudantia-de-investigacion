#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Design matrices for the first-stage income models. The same function is applied to the
CASEN file (estimation) and to the census file (prediction) of a given year, so columns
are identical by construction.

spec 'A' : Agostini-Brown-Román (2010) style list restricted to what the year's census has.
spec 'B' : everything the year's census and CASEN share, plus urban interactions and
           squares (LASSO selects).
spec 'K' : core set common to 1992, 2002 and 2024 (for the historical comparison).
"""
import numpy as np
import pandas as pd

DEMOG = ['n_people', 'n_people_sq', 'n_kids', 'frac_kids', 'n_old', 'frac_old', 'mean_age',
         'head_female', 'head_age', 'head_schooling', 'mean_schooling_adults', 'max_schooling',
         'head_employed', 'n_employed', 'frac_employed_adults', 'dep_ratio', 'head_literate',
         'frac_literate_adults', 'head_indep']
DWELL = ['dw_apt', 'dw_precarious', 'wall_solid', 'wall_precarious', 'roof_good', 'roof_precarious',
         'floor_good', 'floor_dirt', 'water_public', 'water_inside', 'sanit_sewer', 'sanit_none',
         'elec_public', 'elec_none', 'n_bedrooms', 'tenure_owned', 'tenure_rented']
ASSETS = {
    2024: ['a_internet_fixed', 'a_internet_mobile', 'a_internet_sat', 'fuel_wood', 'heat_wood'],
    2017: [],
    1992: ['n_rooms'],
    2002: ['n_rooms', 'a_washer', 'a_fridge', 'a_phone', 'a_video', 'a_microwave', 'a_computer', 'a_calefont', 'a_internet', 'a_cable_tv', 'a_cellphone'],
}
# variables the year's census does not have (dropped from every spec)
UNAVAILABLE = {
    2017: ['head_literate', 'frac_literate_adults', 'head_indep', 'water_inside', 'sanit_sewer', 'sanit_none',
           'elec_public', 'elec_none', 'tenure_owned', 'tenure_rented'],
}
# Agostini et al. (2010) first stage: durable goods, head schooling, female head, fraction of
# children, number of children, household size and its square, roof, rooms, urban, walls, water.
AGOSTINI = {
    2024: ['a_internet_fixed', 'a_internet_mobile', 'a_internet_sat', 'head_schooling', 'head_female',
           'frac_kids', 'n_kids', 'n_people', 'n_people_sq', 'roof_good', 'roof_precarious', 'n_bedrooms',
           'wall_solid', 'wall_precarious', 'water_public', 'water_inside', 'elec_none', 'sanit_sewer',
           'head_employed', 'head_age'],
    1992: ['head_schooling', 'head_female', 'frac_kids', 'n_kids', 'n_people', 'n_people_sq',
           'roof_good', 'roof_precarious', 'n_rooms', 'n_bedrooms', 'wall_solid', 'wall_precarious',
           'water_public', 'water_inside', 'elec_none', 'sanit_sewer', 'head_employed', 'head_age'],
    2017: ['head_schooling', 'head_female', 'frac_kids', 'n_kids', 'n_people', 'n_people_sq',
           'roof_good', 'roof_precarious', 'n_bedrooms', 'wall_solid', 'wall_precarious',
           'water_public', 'head_employed', 'head_age'],
    # Agostini et al. (2010) list as published (internet, washer, calefont, cell/fixed phone, cable TV, microwave, ...)
    2002: ['a_internet', 'a_washer', 'a_calefont', 'a_cellphone', 'a_phone', 'a_cable_tv', 'a_microwave', 'head_schooling', 'head_female',
           'frac_kids', 'n_kids', 'n_people', 'n_people_sq', 'roof_zinc', 'roof_precarious', 'n_rooms', 'wall_solid', 'wall_precarious',
           'water_public', 'water_inside', 'elec_none', 'sanit_sewer', 'head_employed', 'head_age'],
}
CORE_K = DEMOG + DWELL
REGIONS13 = list(range(1, 14))


def spec_vars(year, spec):
    if spec == 'A':
        v = list(AGOSTINI[year])
    elif spec in ('B', 'G'):
        v = DEMOG + DWELL + ASSETS[year]
    elif spec == 'K':
        v = list(CORE_K)
    else:
        raise ValueError(spec)
    return [x for x in v if x not in UNAVAILABLE.get(year, [])]


def design_matrix(df, year, spec, region_col='region13'):
    """Return (X DataFrame, column list). Missing values are imputed with the column
    median of the *given* frame (small share), after which an indicator is added."""
    base = spec_vars(year, spec)
    X = df[base].astype(float).copy()
    # region dummies (13-region system of 1992, drop one)
    reg = df[region_col].astype(float)
    for r in REGIONS13[1:]:
        X[f'reg_{r}'] = (reg == r).astype(float)
    X['urban'] = df['urban'].astype(float)
    if spec in ('B', 'K', 'G'):
        # squares of continuous variables and urban interactions
        cont = [c for c in ['head_age', 'head_schooling', 'mean_schooling_adults', 'mean_age', 'n_bedrooms', 'n_rooms'] if c in X]
        for c in cont:
            X[f'{c}_sq'] = X[c] ** 2
        for c in base:
            X[f'{c}_x_urban'] = X[c] * X['urban']
    # impute
    miss = X.isna()
    for c in X.columns[miss.any()]:
        X[c] = X[c].fillna(X[c].median())
    X['any_missing'] = miss.any(axis=1).astype(float)
    return X

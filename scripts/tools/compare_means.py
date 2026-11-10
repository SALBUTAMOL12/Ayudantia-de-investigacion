#!/usr/bin/env python3
"""Weighted means of the shared variables: census vs CASEN, by year (ELL diagnostic)."""
import os, sys
import numpy as np
import pandas as pd
from scripts.directories import data_inter
from scripts.tools.features import spec_vars

HARM = os.path.join(data_inter, 'harmonized')
year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
c = pd.read_parquet(os.path.join(HARM, f'census_{year}_hh.parquet'))
s = pd.read_parquet(os.path.join(HARM, f'casen_{year}_hh.parquet'))
cols = [v for v in dict.fromkeys(spec_vars(year, 'B') + ['urban']) if v in c and v in s]
rows = []
for v in cols:
    cm = np.average(c[v].dropna(), weights=None)
    sv = s[[v, 'weight']].dropna(); sm = np.average(sv[v], weights=sv.weight)
    rows.append((v, cm, sm, cm - sm, c[v].isna().mean(), (cm - sm) / (sv[v].std() + 1e-9)))
df = pd.DataFrame(rows, columns=['var', 'census', 'casen_w', 'diff', 'census_na', 'diff_sd'])
pd.set_option('display.width', 200)
print(df.round(3).to_string())
print('\nlargest |diff/sd|:'); print(df.reindex(df.diff_sd.abs().sort_values(ascending=False).index).head(12).round(3).to_string())

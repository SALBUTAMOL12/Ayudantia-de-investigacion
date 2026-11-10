#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directory configuration for the eii (ethnic income inequality) project.

Import this module at the top of any script to get consistent paths. It also
sets the working directory to the project root and adds the root to sys.path
so that `from scripts.tools import ...` works from any entry point.

Data layout
-----------
data/
├── A_raw/      # symlinks into ~/Dropbox/_ddbb (populated by configure_data.py)
│   ├── raw/INE/census_1992, census_2002, census_2017, census_2024, Cartografia_censo2024_Pais, dpa_historica
│   ├── raw/MDS/casen         # casen_1992.dta, casen_2003.dta, casen_2017.dta, casen_2024.dta
│   └── raw/BCN/comunas
├── B_inter/    # -> ~/Dropbox/eii/B_inter  (pipeline intermediates, synced)
└── C_final/    # analysis-ready files (tracked in git when small)

Author: Felipe Jordán, Pontificia Universidad Católica de Chile
"""
import os
import sys
from pathlib import Path

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:  # interactive session
    PROJECT_ROOT = str(Path.home() / 'research' / 'eii')

os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

data_raw   = os.path.join(PROJECT_ROOT, 'data', 'A_raw')
data_inter = os.path.join(PROJECT_ROOT, 'data', 'B_inter')
data_final = os.path.join(PROJECT_ROOT, 'data', 'C_final')

_raw = os.path.join(data_raw, 'raw')
data_ine   = os.path.join(_raw, 'INE')
data_mds   = os.path.join(_raw, 'MDS')
data_mdsyf = os.path.join(_raw, 'MDSyF')
data_bcn   = os.path.join(_raw, 'BCN')

# Census and survey files
census_1992 = os.path.join(data_ine, 'census_1992')
census_2002 = os.path.join(data_ine, 'census_2002')
census_2017 = os.path.join(data_ine, 'census_2017')
census_2024 = os.path.join(data_ine, 'census_2024')
carto_2024  = os.path.join(data_ine, 'Cartografia_censo2024_Pais')
dpa_hist    = os.path.join(data_ine, 'dpa_historica')
casen_dir   = os.path.join(data_mds, 'casen')

results_figures = os.path.join(PROJECT_ROOT, 'results', 'figures')
results_tables  = os.path.join(PROJECT_ROOT, 'results', 'tables')
paper_numbers   = os.path.join(PROJECT_ROOT, 'results', 'paper_numbers.tex')

for _d in ['harmonized', 'models', 'simulations', 'groups', 'logs']:
    os.makedirs(os.path.join(data_inter, _d), exist_ok=True)
os.makedirs(data_final, exist_ok=True)
os.makedirs(results_figures, exist_ok=True)
os.makedirs(results_tables, exist_ok=True)

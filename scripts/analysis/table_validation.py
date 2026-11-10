#!/usr/bin/env python3
"""Table: in-sample validation of the simulation designs on the CASEN sample
(official person poverty vs simulated), from B_inter/models/validation.csv."""
import os
import pandas as pd
from scripts.directories import data_inter, results_tables
from scripts.tools.paper_results import write_macro

v = pd.read_csv(os.path.join(data_inter, 'models', 'validation.csv')).sort_values(['year', 'spec'])
lab = {'A': 'ELL: OLS, normal errors', 'B': 'LASSO, bootstrap coefficients, empirical residuals'}
t = ['\\begin{tabular}{llccc}', '\\toprule', 'Year & Design & Official & Simulated & (s.d.) \\\\', '\\midrule']
yr = {2024: 'twentyfour', 1992: 'nineteen', 2002: 'twothousandtwo', 2017: 'seventeen'}
for _, r in v.iterrows():
    t.append(f'{int(r.year)} & {lab.get(r.spec, r.spec)} & {100*r.official:.1f} & {100*r.simulated:.1f} & ({100*r.sim_sd:.1f}) \\\\')
    write_macro(f'valid{r.spec}{yr[int(r.year)]}', 100 * r.simulated); write_macro(f'validOfficial{yr[int(r.year)]}', 100 * r.official)
t += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(results_tables, 'validation_insample.tex'), 'w').write('\n'.join(t) + '\n')
print('\n'.join(t))

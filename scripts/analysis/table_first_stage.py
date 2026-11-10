#!/usr/bin/env python3
"""Table: fit of the first-stage income models (from B_inter/models/metrics.csv)."""
import os
import pandas as pd
from scripts.directories import data_inter, results_tables
from scripts.tools.paper_results import write_macro

m = pd.read_csv(os.path.join(data_inter, 'models', 'metrics.csv'))
lab = {'A': 'OLS, Agostini et al. (2010) list (ELL)', 'B': 'LASSO, all shared variables', 'K': 'LASSO, core set common to all censuses', 'G': 'Gradient boosting, all shared variables'}
t = ['\\begin{tabular}{llcccc}', '\\toprule', 'Year & First stage & $k$ & $R^2$ train & $R^2$ held-out comunas & $\\sigma_\\varepsilon$ \\\\', '\\midrule']
for _, r in m.sort_values(['year', 'spec']).iterrows():
    t.append(f'{int(r.year)} & {lab[r.spec]} & {int(r.n_nonzero)} & {r.r2_train:.3f} & {r.r2_test:.3f} & {r.sig_eps:.3f} \\\\')
t += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(results_tables, 'first_stage_fit.tex'), 'w').write('\n'.join(t) + '\n')
yr = {2024: 'twentyfour', 1992: 'nineteen', 2002: 'twothousandtwo', 2017: 'seventeen'}
for _, r in m.iterrows():
    write_macro(f'rtwo{r.spec}{yr[int(r.year)]}', r.r2_test, '{:.2f}')
print('\n'.join(t))

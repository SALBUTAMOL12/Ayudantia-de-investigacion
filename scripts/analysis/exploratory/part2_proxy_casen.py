#!/usr/bin/env python3
"""Gradient of OBSERVED survey poverty (CASEN 2017, 2024) on the 1992-model Mapuche probability of the
respondent (inputs: mother's comuna, 5-year cohort, sex; cohorts 1914-1978), compared with the gradient on
contemporaneous self-identification. Person-level WLS with survey weights; (i) no FE, (ii) comuna-of-residence FE.
SE clustered by comuna of residence. No imputation involved."""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scripts.directories import data_inter, results_tables
from scripts.tools.paper_results import write_macro
from scripts.analysis.exploratory.part2_proxy_logit import fit_logit_1992

HARM = os.path.join(data_inter, 'harmonized')
predict, _ = fit_logit_1992(False)
rows = []
for y in [2017, 2024]:
    per = pd.read_parquet(os.path.join(HARM, f'casen_{y}_per.parquet'))
    hh = pd.read_parquet(os.path.join(HARM, f'casen_{y}_hh.parquet'), columns=['hh_id', 'poor_official'])
    d = per.merge(hh, on='hh_id')
    d['birth_year'] = y - d.age; d['cohort5'] = (d.birth_year // 5) * 5
    d = d[(d.birth_year >= 1914) & (d.birth_year <= 1978) & d.mother_comuna.notna() & d.mapuche.notna()].copy()
    cells = d[['cohort5', 'female', 'mother_comuna']].drop_duplicates()
    cells['phat'] = predict(cells).to_numpy()
    d = d.merge(cells, on=['cohort5', 'female', 'mother_comuna'])
    for xv, lab in [('phat', 'proxy'), ('mapuche', 'selfid')]:
        for fe in [False, True]:
            X = d[[xv]].astype(float)
            if fe:
                X = pd.concat([X, pd.get_dummies(d.comuna.astype(int), drop_first=True, dtype=float)], axis=1)
            else:
                X = sm.add_constant(X)
            r = sm.WLS(d.poor_official.to_numpy(), X.to_numpy(), weights=d.weight.to_numpy()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.comuna)[0]})
            rows.append(dict(year=y, x=lab, fe=fe, b=r.params[0 if fe else 1], se=r.bse[0 if fe else 1], n=len(d)))
    print(y, 'n', len(d), 'mean phat', d.phat.mean().round(3), 'mean selfid', np.average(d.mapuche, weights=d.weight).round(3))
t = pd.DataFrame(rows); print(t.round(3).to_string())
tex = ['\\begin{tabular}{llcc}', '\\toprule', 'Year & Regressor & No fixed effects & Comuna-of-residence FE \\\\', '\\midrule']
for y in [2017, 2024]:
    for xv, lab in [('proxy', '1992-model Mapuche probability'), ('selfid', 'Self-identifies as Mapuche')]:
        a = t[(t.year == y) & (t.x == xv) & (~t.fe)].iloc[0]; b = t[(t.year == y) & (t.x == xv) & (t.fe)].iloc[0]
        tex.append(f'{y} & {lab} & {100*a.b:.1f} ({100*a.se:.1f}) & {100*b.b:.1f} ({100*b.se:.1f}) \\\\')
        yn = {2017: 'Seventeen', 2024: 'TwentyFour'}[y]
        write_macro(f'casen{xv.capitalize()}Grad{yn}', 100 * a.b); write_macro(f'casen{xv.capitalize()}Grad{yn}Se', 100 * a.se)
        write_macro(f'casen{xv.capitalize()}GradFE{yn}', 100 * b.b); write_macro(f'casen{xv.capitalize()}GradFE{yn}Se', 100 * b.se)
tex += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(results_tables, 'casen_proxy_gradient.tex'), 'w').write('\n'.join(tex) + '\n'); print('\n'.join(tex))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicted-probability (proxy) version of Part 2. A logit of Mapuche self-identification is fitted on the 1992
census using only predetermined characteristics: comuna of origin (fixed effects), 5-year cohort, sex and,
in the second version, the schooling bracket (0-8, 9-12, 13+) for cohorts aged 25+ in 1992 (born <= 1967),
whose schooling was complete. The fitted model gives every person in every census a probability p_hat from the
same inputs, so the proxy cannot drift with self-identification. Cell poverty (cells = origin x sex x cohort
[x schooling bracket]) is then regressed on p_hat in the contiguous-pair design with pair-by-year effects;
in the schooling version the outcome regression also controls for schooling-bracket-by-year effects, so the
gradient is identified from the interaction pattern and not from the direct effect of schooling on poverty.

Usage: python -m scripts.analysis.part2_proxy_logit [--spec B] [--threshold 5pp]
"""
import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scripts.directories import data_inter, data_final, results_tables
from scripts.tools.paper_results import write_macro
from scripts.analysis.part2_pair_did import absorb_two_way, fmt

HARM = os.path.join(data_inter, 'harmonized'); SIM = os.path.join(data_inter, 'simulations')
YEARS = [1992, 2002, 2017, 2024]


def fit_logit_1992(with_schooling):
    per = pd.read_parquet(os.path.join(HARM, 'census_1992_per.parquet'), columns=['cohort5', 'female', 'mother_comuna', 'mapuche', 'schooling', 'birth_year'])
    per = per[(per.birth_year >= 1914) & (per.birth_year <= (1967 if with_schooling else 1978)) & per.mapuche.notna() & per.mother_comuna.notna()]
    per['sb'] = pd.cut(per.schooling, [-1, 8, 12, 99], labels=False).astype(float)
    if with_schooling:
        per = per[per.sb.notna()]
    keys = ['cohort5', 'female', 'mother_comuna'] + (['sb'] if with_schooling else [])
    g = per.groupby(keys).mapuche.agg(['sum', 'count']).reset_index()
    # logit with comuna FE + cohort + sex (+ schooling): fit on cell counts (binomial), additive on the logit scale
    X = pd.concat([pd.get_dummies(g.mother_comuna.astype(int), prefix='m', dtype=float),
                   pd.get_dummies(g.cohort5.astype(int), prefix='c', drop_first=True, dtype=float),
                   g[['female']].astype(float)] + ([pd.get_dummies(g.sb.astype(int), prefix='s', drop_first=True, dtype=float)] if with_schooling else []), axis=1)
    y = np.column_stack([g['sum'], g['count'] - g['sum']])
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    cols = list(X.columns)
    def predict(cells):
        Xp = pd.concat([pd.get_dummies(cells.mother_comuna.astype(int), prefix='m', dtype=float),
                        pd.get_dummies(cells.cohort5.astype(int), prefix='c', drop_first=True, dtype=float),
                        cells[['female']].astype(float)] + ([pd.get_dummies(cells.sb.astype(int), prefix='s', drop_first=True, dtype=float)] if with_schooling else []), axis=1)
        Xp = Xp.reindex(columns=cols, fill_value=0.0)
        return r.predict(Xp)
    return predict, r


def main(spec, threshold):
    pairs = pd.read_parquet(os.path.join(data_final, 'comuna_pairs.parquet'))
    pairs = pairs[{'01sd': pairs.keep_01sd, '5pp': pairs['diff'] >= 0.05, '10pp': pairs['diff'] >= 0.10}[threshold]]
    out_rows = []
    for with_schooling in [False, True]:
        predict, r = fit_logit_1992(with_schooling)
        parts = []
        for y in YEARS:
            f = os.path.join(SIM, f'{y}_{spec}_' + ('cells_sch' if with_schooling else 'groups') + '.parquet')
            if os.path.exists(f):
                d = pd.read_parquet(f); d['year'] = y; parts.append(d)
        if len(parts) < 3:
            print('not enough years for', with_schooling); continue
        d = pd.concat(parts)
        d = d[(d.cohort5 >= 1914) & (d.cohort5 <= (1965 if with_schooling else 1975))]
        d['phat'] = predict(d).to_numpy()
        d['comuna'] = d.mother_comuna.astype(int)
        rows = []
        for _, p in pairs.iterrows():
            for col in ['comuna_a', 'comuna_b']:
                s = d[d.comuna == int(p[col])].copy(); s['pair'] = p.pair_id; rows.append(s)
        st = pd.concat(rows, ignore_index=True); st = st[st.poor.notna() & (st.n >= 5)]
        years = sorted(st.year.unique())
        X = np.column_stack([(st.phat * (st.year == y_)).to_numpy() for y_ in years])
        w = st.n.to_numpy(float)
        f1 = (st.pair.astype(str) + '_' + st.year.astype(str)).to_numpy()
        f2 = ((st.sb.astype(int).astype(str) if with_schooling else st.female.astype(int).astype(str)) + '_' + st.year.astype(str)).to_numpy()
        yt, Xt = absorb_two_way(st.poor.to_numpy(), X, w, f1, f2)
        rr = sm.WLS(yt, Xt, weights=w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(st.comuna)[0]})
        V = rr.cov_params()
        res = {f'b{y_}': [rr.params[i], rr.bse[i], np.nan] for i, y_ in enumerate(years)}
        for i, y_ in enumerate(years[1:], start=1):
            res[f'd{y_}'] = [rr.params[i] - rr.params[0], np.sqrt(V[i, i] + V[0, 0] - 2 * V[i, 0]), np.nan]
        res['n'] = len(st); res['pairs'] = st.pair.nunique()
        out_rows.append((with_schooling, years, res))
        print('with schooling' if with_schooling else 'without schooling', {k: (round(100 * v[0], 1), round(100 * v[1], 1)) for k, v in res.items() if k[0] in 'bd'}, 'cells', len(st))
    tex = ['\\begin{tabular}{lcc}', '\\toprule', ' & Proxy: origin, cohort, sex & Proxy adds schooling (cohorts born $\\le$1967) \\\\', '\\midrule']
    yrs_all = sorted({y for _, yrs, _ in out_rows for y in yrs})
    for y_ in yrs_all:
        tex.append(f'$\\beta_{{{y_}}}$ & ' + ' & '.join(fmt(res[f'b{y_}']) if f'b{y_}' in res else '--' for _, _, res in out_rows) + ' \\\\')
    tex.append('\\midrule')
    for y_ in yrs_all[1:]:
        tex.append(f'$\\beta_{{{y_}}}-\\beta_{{{yrs_all[0]}}}$ & ' + ' & '.join(fmt(res[f'd{y_}']) if f'd{y_}' in res else '--' for _, _, res in out_rows) + ' \\\\')
    tex.append('\\midrule'); tex.append('Cells & ' + ' & '.join(str(res['n']) for _, _, res in out_rows) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, f'pair_did_proxy_{spec}_{threshold}.tex'), 'w').write('\n'.join(tex) + '\n'); print('\n'.join(tex))
    yn = {1992: 'NinetyTwo', 2002: 'TwoThousandTwo', 2017: 'Seventeen', 2024: 'TwentyFour'}
    for ws, yrs, res in out_rows:
        for k, v in res.items():
            if k[0] in 'bd':
                nm = 'proxy' + ('Sch' if ws else '') + ('Beta' if k[0] == 'b' else 'Diff') + yn[int(k[1:])]
                write_macro(nm, 100 * v[0]); write_macro(nm + 'Se', 100 * v[1])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', default='B'); ap.add_argument('--threshold', default='5pp'); a = ap.parse_args()
    main(a.spec, a.threshold)

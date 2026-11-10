#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 1: poverty by ethnicity, 2024. Columns: (1) direct CASEN 2024 estimates with
survey-design bootstrap SEs; (2) ELL/Agostini-style first stage (spec A) applied to the
census; (3) LASSO first stage (spec B); the EO-ML column is a placeholder documented in
the paper (2017 pilot). Panels: urban, rural, national. Also the first-stage fit table.

Usage: python -m scripts.analysis.table1_poverty_2024 [--year 2024]
"""
import argparse
import os
import numpy as np
import pandas as pd
from scripts.directories import data_inter, results_tables, casen_dir
from scripts.tools.paper_results import write_macro

HARM = os.path.join(data_inter, 'harmonized'); SIM = os.path.join(data_inter, 'simulations')
PUEBLO = {2024: {0: 'Non-indigenous', 1: 'Mapuche', 2: 'Aymara', 3: 'Rapa Nui', 4: 'Lickanantay', 5: 'Quechua', 6: 'Colla', 7: 'Diaguita', 8: 'Kaw\\\'esqar', 9: 'Yag\\\'an', 10: 'Chango', 11: 'Selk\\\'nam', 12: 'Other'},
          1992: {0: 'Non-indigenous', 1: 'Mapuche', 2: 'Aymara', 3: 'Rapa Nui'},
          2002: {0: 'Non-indigenous', 1: 'Mapuche', 2: 'Aymara', 3: 'Rapa Nui', 4: 'Atacame\\~no', 5: 'Quechua', 6: 'Colla', 8: 'Kaw\\\'esqar', 9: 'Yag\\\'an'},
          2017: {0: 'Non-indigenous', 1: 'Mapuche', 2: 'Aymara', 3: 'Rapa Nui', 4: 'Lickanantay', 5: 'Quechua', 6: 'Colla', 7: 'Diaguita', 8: 'Kaw\\\'esqar', 9: 'Yag\\\'an', 12: 'Other'}}
CASEN_R3 = {1: 2, 2: 3, 3: 5, 4: 1, 5: 4, 6: 6, 7: 8, 8: 9, 9: 7, 10: 10, 11: 11}  # casen r3 -> census pueblo code


def direct_casen(year, B=200, seed=3):
    """Person-level poverty by ethnicity x area from CASEN, cluster bootstrap by varunit."""
    cols = ['folio', 'expr', 'varunit', 'area', 'pobreza', 'pueblos_indigenas', 'r3', 'edad']
    if year == 2024:
        d = pd.read_stata(os.path.join(casen_dir, 'casen_2024.dta'), convert_categoricals=False, columns=cols)
        d['poor'] = d.pobreza.isin([1, 2]).astype(float); d['urban'] = (d.area == 1).astype(int)
        d['ind'] = np.where(d.pueblos_indigenas == 1, 1, np.where(d.pueblos_indigenas == 0, 0, 2))
        d['pueblo'] = np.where(d.ind == 1, d.r3.map(CASEN_R3).fillna(12), 0).astype(int)
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(d.varunit); idx_by = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    def rates(dd):
        out = {}
        for u in [1, 0, None]:
            s = dd if u is None else dd[dd.urban == u]
            for e in [0, 1]:
                ss = s[s.ind == e]; out[(u, 'eth', e)] = np.average(ss.poor, weights=ss.expr)
        for p in PUEBLO[year]:
            ss = dd[dd.pueblo == p]
            out[(None, 'pueblo', p)] = np.average(ss.poor, weights=ss.expr) if len(ss) else np.nan
        return pd.Series(out)
    est = rates(d)
    boots = []
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq)); idx = np.concatenate([idx_by[k] for k in pick])
        boots.append(rates(d.iloc[idx]))
    sd = pd.concat(boots, axis=1).std(axis=1)
    return est, sd


def sim_cells(year, spec):
    c = pd.read_csv(os.path.join(SIM, f'{year}_{spec}_cells.csv'))
    out = {}
    e = c[c.cell == 'eth_x_urban'].set_index('code')
    for u in [1, 0]:
        for eth in [0, 1]:
            k = eth * 2 + u; out[(u, 'eth', eth)] = (e.loc[k, 'poor'], e.loc[k, 'poor_sd'])
    # national by ethnicity: combine urban+rural cells (weighted by n)
    for eth in [0, 1]:
        ks = [eth * 2 + u for u in [0, 1]]; n = e.loc[ks, 'n']
        out[(None, 'eth', eth)] = (np.average(e.loc[ks, 'poor'], weights=n), np.sqrt(np.sum((n / n.sum()) ** 2 * e.loc[ks, 'poor_sd'] ** 2)))
    p = c[c.cell == 'pueblo'].set_index('code')
    for k in PUEBLO[year]:
        if k in p.index:
            out[(None, 'pueblo', k)] = (p.loc[k, 'poor'], p.loc[k, 'poor_sd'])
    return out


def fmt(x, sd):
    return f'{100*x:.1f} ({100*sd:.1f})' if pd.notna(x) else '--'


def main(year):
    if year == 2024:
        est, sd = direct_casen(year)
    else:   # CASEN 1992 did not ask about ethnicity
        est, sd = pd.Series(dtype=float), pd.Series(dtype=float)
    A, B = sim_cells(year, 'A'), sim_cells(year, 'B')
    rows = []
    for u, lab in [(1, 'Urban'), (0, 'Rural'), (None, 'National')]:
        rows.append((f'\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\', None))
        for eth, name in [(0, 'Non-indigenous'), (1, 'Indigenous')]:
            k = (u, 'eth', eth)
            rows.append((f'\\quad {name} & {fmt(est.get(k, np.nan), sd.get(k, np.nan))} & {fmt(*A[k])} & {fmt(*B[k])} \\\\', None))
    rows.append(('\\multicolumn{4}{l}{\\textit{National, by people}} \\\\', None))
    for p, name in PUEBLO[year].items():
        if p == 0:
            continue
        k = (None, 'pueblo', p)
        if k in A:
            rows.append((f'\\quad {name} & {fmt(est.get(k, np.nan), sd.get(k, np.nan))} & {fmt(*A[k])} & {fmt(*B[k])} \\\\', None))
    tex = ['\\begin{tabular}{lccc}', '\\toprule',
           ' & CASEN direct & ELL (Agostini) & LASSO \\\\', ' & (1) & (2) & (3) \\\\', '\\midrule']
    tex += [r for r, _ in rows]
    tex += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, f'poverty_by_ethnicity_{year}.tex'), 'w').write('\n'.join(tex) + '\n')
    # appendix: head-based ethnicity (household classified by the head's self-identification)
    hb = []
    for spec in ['A', 'B']:
        c = pd.read_csv(os.path.join(SIM, f'{year}_{spec}_cells.csv')); e = c[c.cell == 'head_eth_x_urban'].set_index('code')
        for u, lab in [(1, 'Urban'), (0, 'Rural')]:
            for eth, nm in [(0, 'Non-indigenous head'), (1, 'Indigenous head')]:
                k = eth * 2 + u; hb.append((lab, nm, spec, e.loc[k, 'poor'], e.loc[k, 'poor_sd']))
    hb = pd.DataFrame(hb, columns=['area', 'eth', 'spec', 'poor', 'sd'])
    t2 = ['\\begin{tabular}{llcc}', '\\toprule', ' & & ELL (Agostini) & LASSO \\\\', '\\midrule']
    for (area, eth), sub in hb.groupby(['area', 'eth'], sort=False):
        a_ = sub[sub.spec == 'A'].iloc[0]; b_ = sub[sub.spec == 'B'].iloc[0]
        t2.append(f'{area} & {eth} & {fmt(a_.poor, a_.sd)} & {fmt(b_.poor, b_.sd)} \\\\')
    t2 += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, f'poverty_by_head_ethnicity_{year}.tex'), 'w').write('\n'.join(t2) + '\n')
    # metrics table
    m = pd.read_csv(os.path.join(data_inter, 'models', 'metrics.csv'))
    lab = {'A': 'OLS, Agostini variable list (ELL)', 'B': 'LASSO, all shared variables', 'K': 'LASSO, core set common to all censuses', 'G': 'Gradient boosting, all shared variables'}
    t = ['\\begin{tabular}{llcccc}', '\\toprule', 'Year & First stage & $k$ & $R^2$ train & $R^2$ held-out comunas & $\\sigma_\\varepsilon$ \\\\', '\\midrule']
    for _, r in m.sort_values(['year', 'spec']).iterrows():
        t.append(f'{int(r.year)} & {lab[r.spec]} & {int(r.n_nonzero)} & {r.r2_train:.3f} & {r.r2_test:.3f} & {r.sig_eps:.3f} \\\\')
    t += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, 'first_stage_fit.tex'), 'w').write('\n'.join(t) + '\n')
    yr = {2024: 'twentyfour', 1992: 'nineteen', 2017: 'seventeen', 2002: 'twothousandtwo'}[year]
    for eth, nm in [(0, 'NonInd'), (1, 'Ind')]:
        k = (None, 'eth', eth)
        write_macro(f'povA{nm}{yr}', 100 * A[k][0]); write_macro(f'povB{nm}{yr}', 100 * B[k][0])
        if k in est: write_macro(f'povCasen{nm}{yr}', 100 * est[k])
    gA = A[(None, 'eth', 1)][0] - A[(None, 'eth', 0)][0]; gB = B[(None, 'eth', 1)][0] - B[(None, 'eth', 0)][0]
    write_macro(f'gapA{yr}', 100 * gA); write_macro(f'gapB{yr}', 100 * gB)
    if (None, 'eth', 1) in est: write_macro(f'gapCasen{yr}', 100 * (est[(None, 'eth', 1)] - est[(None, 'eth', 0)]))
    for u, nm in [(1, 'Urb'), (0, 'Rur')]:
        for eth, en in [(0, 'NonInd'), (1, 'Ind')]:
            k = (u, 'eth', eth); write_macro(f'pov{nm}{en}A{yr}', 100 * A[k][0]); write_macro(f'pov{nm}{en}B{yr}', 100 * B[k][0])
            if k in est: write_macro(f'pov{nm}{en}Casen{yr}', 100 * est[k])
    print('\n'.join(tex)); print('\n'.join(t))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--year', type=int, default=2024)
    main(ap.parse_args().year)

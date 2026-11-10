#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2, ecological panel. Cells g = (5-year cohort c, sex s, mother's comuna m); years 1992, 2017, 2024.
Outcome Y_gt = imputed poverty (first stage spec B of each year) or the self-identification share.

Spec 1 (broad):      Y_gt = alpha_ms + gamma_cs + lambda_t + sum_t beta_t X_g 1[t] + e
Spec 2 (saturated):  Y_gt = alpha_mst + gamma_cst         + sum_t beta_t X_g 1[t] + e
X_g = 1992 indigenous share of the cell (fixed). Weights = 1992 cell population. SE clustered by m.

Outputs: results/tables/gap_panel_{spec}.tex, macros, and results/figures/gap_panel.png.
Usage: python -m scripts.analysis.part2_gap_panel [--spec B] [--min-n 20]
"""
import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.directories import data_final, data_inter, results_tables, results_figures
from scripts.tools.paper_results import write_macro

SIM = os.path.join(data_inter, 'simulations')
YEARS = [1992, 2017, 2024]


def load_panel(spec, min_n, share='indigenous', cohorts=(1914, 1975)):
    g = pd.read_parquet(os.path.join(data_final, 'groups_panel.parquet'))
    g = g[(g.cohort5 >= cohorts[0]) & (g.cohort5 <= cohorts[1])]
    base = g[g.year == 1992][['cohort5', 'female', 'mother_comuna', 'n', share]].rename(columns={'n': 'w', share: 'X'})
    base['X'] = base.X.fillna(0)
    g = g.merge(base, on=['cohort5', 'female', 'mother_comuna'])
    ok = g.groupby(['cohort5', 'female', 'mother_comuna']).year.transform('nunique') == len([y for y in YEARS if y in g.year.unique()])
    g = g[ok & (g.n >= min_n)].copy()
    cnt = g.groupby(['cohort5', 'female', 'mother_comuna']).year.transform('size')
    g = g[cnt == g.year.nunique()].copy()
    g['pov'] = g[f'poor_{spec}'] if f'poor_{spec}' in g else np.nan
    g['selfid'] = g['mapuche']
    g['ms'] = g.mother_comuna.astype(int).astype(str) + '_' + g.female.astype(int).astype(str)
    g['cs'] = g.cohort5.astype(int).astype(str) + '_' + g.female.astype(int).astype(str)
    g['t'] = g.year.astype(int).astype(str)
    return g


def fe_regression(g, y, saturated):
    years = sorted(g.year.unique())
    X = pd.DataFrame({f'b{y_}': g.X * (g.year == y_) for y_ in years}, index=g.index)
    if saturated:
        D = pd.concat([pd.get_dummies(g.ms + '_' + g.t, dtype=float), pd.get_dummies(g.cs + '_' + g.t, dtype=float, drop_first=True)], axis=1)
    else:
        D = pd.concat([pd.get_dummies(g.ms, dtype=float), pd.get_dummies(g.cs, dtype=float, drop_first=True), pd.get_dummies(g.t, dtype=float, drop_first=True)], axis=1)
    Xf = pd.concat([X, D], axis=1)
    m = g[y].notna()
    r = sm.WLS(g.loc[m, y].to_numpy(), Xf.loc[m].to_numpy(), weights=g.loc[m, 'w'].to_numpy()).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(g.loc[m, 'mother_comuna'])[0]})
    out = {f'b{y_}': (r.params[i], r.bse[i]) for i, y_ in enumerate(years)}
    # differences relative to 1992
    V = r.cov_params()
    for i, y_ in enumerate(years[1:], start=1):
        d = r.params[i] - r.params[0]; se = np.sqrt(V[i, i] + V[0, 0] - 2 * V[i, 0])
        out[f'd{y_}'] = (d, se)
    out['n'] = int(m.sum()); out['cells'] = int(g.loc[m].groupby(['cohort5', 'female', 'mother_comuna']).ngroups)
    return out


def between_regression(g, y):
    """Collapse to comuna-of-origin x sex x year (1992-population weights); regress on the comuna-sex 1992
    Mapuche share x year with comuna-sex and year FE. Identified from across-comuna trends only."""
    d = g.copy(); d['wy'] = d.w * d[y]; d['wX'] = d.w * d.X
    c = d.groupby(['ms', 'year', 'mother_comuna']).agg(wy=('wy', 'sum'), wX=('wX', 'sum'), w=('w', 'sum')).reset_index()
    c['Y'] = c.wy / c.w; c['Xc'] = c.wX / c.w; c['t'] = c.year.astype(int).astype(str)
    years = sorted(c.year.unique())
    X = pd.DataFrame({f'b{y_}': c.Xc * (c.year == y_) for y_ in years}, index=c.index)
    D = pd.concat([pd.get_dummies(c.ms, dtype=float), pd.get_dummies(c.t, dtype=float, drop_first=True)], axis=1)
    Xf = pd.concat([X, D], axis=1)
    r = sm.WLS(c.Y.to_numpy(), Xf.to_numpy(), weights=c.w.to_numpy()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(c.mother_comuna)[0]})
    out = {f'b{y_}': (r.params[i], r.bse[i]) for i, y_ in enumerate(years)}
    V = r.cov_params()
    for i, y_ in enumerate(years[1:], start=1):
        out[f'd{y_}'] = (r.params[i] - r.params[0], np.sqrt(V[i, i] + V[0, 0] - 2 * V[i, 0]))
    out['n'] = len(c); out['cells'] = int(c.groupby(['ms']).ngroups)
    return out


def fmt(v):
    return f'{100*v[0]:.1f} ({100*v[1]:.1f})'


def main(spec, min_n, share='mapuche', cohorts=(1914, 1975), tag='', measure='mapuche'):
    g = load_panel(spec, min_n, share, cohorts)
    micro_cell = 'mp_x_cohort' if measure == 'mapuche' else 'eth_x_cohort'
    years = sorted(g.year.unique())
    res = {}
    for y in ['pov']:
        if g[y].notna().sum() == 0:
            continue
        for sat in [False, True]:
            res[(y, sat)] = fe_regression(g, y, sat)
        res[(y, 'between')] = between_regression(g[g[y].notna()], y)
    if not res:
        print('no imputed poverty for all years yet'); return
    # micro gaps by self-identification (same cohorts) from simulation cells
    micro = {}
    for y_ in years:
        f = os.path.join(SIM, f'{y_}_{spec}_cells.csv')
        if os.path.exists(f):
            c = pd.read_csv(f); c = c[c.cell == micro_cell].set_index('code')
            if {1, 3}.issubset(c.index):
                micro[y_] = (c.loc[3, 'poor'] - c.loc[1, 'poor'], np.sqrt(c.loc[3, 'poor_sd'] ** 2 + c.loc[1, 'poor_sd'] ** 2))
    # ---- table
    tex = ['\\begin{tabular}{lccc}', '\\toprule', ' & Within comuna & Between comunas & Broad (mixture) \\\\', ' & (saturated FE) & (comuna level) & \\\\', '\\midrule']
    ORDER = [True, 'between', False]
    def cell(y, sat, k):
        r = res.get((y, sat)); return fmt(r[k]) if r and k in r else '--'
    for y_ in years:
        tex.append(f'$\\beta_{{{y_}}}$ & ' + ' & '.join(cell('pov', sat, f'b{y_}') for sat in ORDER) + ' \\\\')
    tex.append('\\midrule')
    for y_ in years[1:]:
        tex.append(f'$\\beta_{{{y_}}}-\\beta_{{1992}}$ & ' + ' & '.join(cell('pov', sat, f'd{y_}') for sat in ORDER) + ' \\\\')
    tex.append('\\midrule')
    for y_ in years:
        if y_ in micro:
            tex.append(f'Micro gap by self-identification, {y_} & \\multicolumn{{3}}{{c}}{{{fmt(micro[y_])}}} \\\\')
    r0 = res[('pov', False)]; rb = res[('pov', 'between')]
    tex.append(f'Units & {r0["cells"]} cells & {rb["cells"]} comuna-sex & {r0["cells"]} cells \\\\')
    tex.append('Fixed effects & $m\\!\\times\\! s\\!\\times\\! t$, $c\\!\\times\\! s\\!\\times\\! t$ & $m\\!\\times\\! s$, $t$ & $m\\!\\times\\! s$, $c\\!\\times\\! s$, $t$ \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    tex = [t for t in tex if t != '']
    open(os.path.join(results_tables, f'gap_panel_{spec}{tag}.tex'), 'w').write('\n'.join(tex) + '\n')
    print('\n'.join(tex))
    # ---- macros
    yn = {1992: 'NinetyTwo', 2017: 'Seventeen', 2024: 'TwentyFour'}
    for (y, sat), r in res.items():
        for k, v in r.items():
            if k in ('n', 'cells'):
                continue
            nm = 'pov' + {True: 'Sat', False: 'Broad', 'between': 'Between'}[sat] + ('Beta' if k[0] == 'b' else 'Diff') + yn[int(k[1:])]
            write_macro(nm + tag, 100 * v[0]); write_macro(nm + tag + 'Se', 100 * v[1])
    for y_, v in micro.items():
        write_macro('microGap' + yn[y_], 100 * v[0]); write_macro('microGap' + yn[y_] + 'Se', 100 * v[1])
    write_macro('panelCells' + tag, r0['cells'], '{:d}')
    # ---- figure: beta_t paths
    fig, axs = plt.subplots(1, 1, figsize=(5, 3.8)); ax = [axs]
    for j, y in enumerate(['pov']):
        for sat, c, lab, off in [(True, '#c0504d', 'Within comuna (saturated FE)', -2), ('between', '#1f4e79', 'Between comunas', 0), (False, '#7f7f7f', 'Broad (mixture)', 2)]:
            r = res.get((y, sat))
            if r is None:
                continue
            b = [100 * r[f'b{y_}'][0] for y_ in years]; se = [100 * r[f'b{y_}'][1] for y_ in years]
            ax[j].errorbar(np.array(years) + off, b, yerr=[1.96 * s for s in se], fmt='o-', color=c, label=lab, capsize=3)
        if y == 'pov' and micro:
            ax[j].plot([y_ for y_ in years if y_ in micro], [100 * micro[y_][0] for y_ in years if y_ in micro], 's--', color='grey', label='Micro gap by self-ID')
        ax[j].set_title('Imputed poverty gap'); ax[j].set_ylabel('coefficient on 1992 share (pp)')
        ax[j].set_xticks(years); ax[j].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(results_figures, f'gap_panel{tag}.png'), dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', default='B'); ap.add_argument('--min-n', type=int, default=20)
    ap.add_argument('--sample', choices=['adults', 'young', 'younghh'], default='adults')
    # adults: own 1992 self-ID, cohorts 1914-1978 (asked in 1992). young: own self-ID, cohorts 1951-1978.
    # younghh: lives with an indigenous adult in 1992, cohorts 1951-1991 (includes children not asked in 1992).
    ap.add_argument('--measure', choices=['mapuche', 'indigenous'], default='mapuche')
    a = ap.parse_args()
    own, hh_ = ('mapuche', 'mapuche_hh') if a.measure == 'mapuche' else ('indigenous', 'indig_hh')
    cfg = {'adults': (own, (1914, 1975), ''), 'young': (own, (1950, 1975), 'Young'), 'younghh': (hh_, (1950, 1990), 'YoungHH')}[a.sample]
    tag = cfg[2] + ('' if a.measure == 'mapuche' else 'Ind')
    main(a.spec, a.min_n, cfg[0], cfg[1], tag, measure=a.measure)

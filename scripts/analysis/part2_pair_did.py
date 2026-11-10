#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2: contiguous-pair difference-in-differences around the 1993 Indigenous Law.

Units: comunas of origin (mother's comuna, 1992 partition), collapsed from the closed cells of the cohorts
asked in 1992 (born 1914-1978), observed in 1992 (pre-Law), 2017 and 2024. Pairs: adjacent comunas whose
1992 Mapuche shares differ by at least the threshold (scripts/tools/comuna_pairs.py). Stacked specification

    y_{ipt} = sum_t beta_t 1[t] * Share_i + theta_{pt} + e_{ipt},

pair-by-year fixed effects, no omitted year (each beta_t is the within-pair gradient of y on the 1992 Mapuche
share in year t); beta_t - beta_1992 is the difference-in-differences. SE clustered by comuna of origin
(each comuna enters several pairs); pair-clustered SE reported as well.

Outcomes: imputed poverty (spec B), Mapuche self-identification share (descriptive), schooling.
Usage: python -m scripts.analysis.part2_pair_did [--spec B] [--threshold 01sd|5pp|10pp] [--weighted]
"""
import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.directories import data_final, results_tables, results_figures
from scripts.tools.paper_results import write_macro

YEARS = [1992, 2017, 2024]


def comuna_panel(spec, cohorts=(1914, 1975)):
    g = pd.read_parquet(os.path.join(data_final, 'groups_panel.parquet'))
    g = g[(g.cohort5 >= cohorts[0]) & (g.cohort5 <= cohorts[1])].copy()
    pc = f'poor_{spec}'
    g['pov'] = g[pc] if pc in g else np.nan
    base = g[g.year == 1992][['cohort5', 'female', 'mother_comuna', 'n']].rename(columns={'n': 'w'})
    g = g.merge(base, on=['cohort5', 'female', 'mother_comuna'])
    def agg(d):
        out = {'pop92': d.w.sum(), 'n': d.n.sum()}
        for c in ['pov', 'mapuche', 'indigenous', 'schooling', 'urban', 'employed', 'floor_dirt', 'sanit_none', 'elec_none', 'water_public', 'wall_precarious', 'persons_per_bedroom', 'hs', 'deprived']:
            if c not in d: continue
            m = d[c].notna()
            out[c] = np.average(d.loc[m, c], weights=d.loc[m, 'w']) if m.any() else np.nan
        return pd.Series(out)
    c = g.groupby(['mother_comuna', 'year']).apply(agg, include_groups=False).reset_index()
    c['comuna'] = c.mother_comuna.astype(int)
    return c


def stack(c, pairs):
    base = c[c.year == 1992].set_index('comuna')
    for cc, src in [('urban92', 'urban'), ('hs92', 'hs'), ('deprived92', 'deprived')]:
        if src in base:
            c = c.merge(base[[src]].rename(columns={src: cc}), left_on='comuna', right_index=True, how='left')
    rows = []
    for _, p in pairs.iterrows():
        for col, share in [('comuna_a', p.share_a), ('comuna_b', p.share_b)]:
            s = c[c.comuna == int(p[col])].copy(); s['pair'] = p.pair_id; s['share92'] = share
            rows.append(s)
    return pd.concat(rows, ignore_index=True)


CONTROLS = ['urban92', 'hs92', 'deprived92']


def pair_did(d, y, weighted=False, controls=None):
    d = d[d[y].notna()].copy()
    years = sorted(d.year.unique())
    X = pd.DataFrame({f'b{y_}': d.share92 * (d.year == y_) for y_ in years}, index=d.index)
    if controls:   # 1992 comuna characteristics interacted with year (all years; identified within pair)
        for cc in controls:
            for y_ in years:
                X[f'{cc}_{y_}'] = d[cc] * (d.year == y_)
    D = pd.get_dummies(d.pair.astype(str) + '_' + d.year.astype(str), dtype=float)
    Xf = pd.concat([X, D], axis=1).to_numpy()
    w = d.pop92.to_numpy() if weighted else np.ones(len(d))
    out = {}
    for cl, nm in [('comuna', 'c'), ('pair', 'p')]:
        r = sm.WLS(d[y].to_numpy(), Xf, weights=w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cl])[0]})
        V = r.cov_params()
        for i, y_ in enumerate(years):
            out.setdefault(f'b{y_}', [r.params[i]]).append(r.bse[i]) if nm == 'c' else out[f'b{y_}'].append(r.bse[i])
        for i, y_ in enumerate(years[1:], start=1):
            dd = r.params[i] - r.params[0]; se = np.sqrt(V[i, i] + V[0, 0] - 2 * V[i, 0])
            out.setdefault(f'd{y_}', [dd]).append(se) if nm == 'c' else out[f'd{y_}'].append(se)
    out['n'] = len(d); out['pairs'] = d.pair.nunique(); out['comunas'] = d.comuna.nunique()
    return out    # each entry: [coef, se_comuna, se_pair]


def absorb_two_way(y, X, w, f1, f2, tol=1e-9, max_iter=500):
    """Weighted alternating projections: remove fixed effects f1 and f2 from y and the columns of X."""
    M = np.column_stack([y, X]).astype(float)
    c1, c2 = pd.factorize(f1)[0], pd.factorize(f2)[0]
    for _ in range(max_iter):
        M0 = M.copy()
        for c in (c1, c2):
            sw = np.bincount(c, weights=w)
            for j in range(M.shape[1]):
                m = np.bincount(c, weights=w * M[:, j]) / sw
                M[:, j] -= m[c]
        if np.abs(M - M0).max() < tol:
            break
    return M[:, 0], M[:, 1:]


def pair_did_residence(d, y):
    """Units: origin comuna x residence comuna x year, stacked by pair; pair-by-year and residence-by-year FE;
    weights = persons; SE clustered by origin comuna."""
    d = d[d[y].notna()].copy()
    years = sorted(d.year.unique())
    X = np.column_stack([(d.share92 * (d.year == y_)).to_numpy() for y_ in years])
    w = d.n.to_numpy(float)
    f1 = d.pair.astype(str) + '_' + d.year.astype(str); f2 = d.comuna92.astype(int).astype(str) + '_' + d.year.astype(str)
    yt, Xt = absorb_two_way(d[y].to_numpy(), X, w, f1.to_numpy(), f2.to_numpy())
    r = sm.WLS(yt, Xt, weights=w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.comuna)[0]})
    V = r.cov_params(); out = {}
    for i, y_ in enumerate(years):
        out[f'b{y_}'] = [r.params[i], r.bse[i], np.nan]
    for i, y_ in enumerate(years[1:], start=1):
        out[f'd{y_}'] = [r.params[i] - r.params[0], np.sqrt(V[i, i] + V[0, 0] - 2 * V[i, 0]), np.nan]
    out['n'] = len(d); out['pairs'] = d.pair.nunique(); out['comunas'] = d.comuna.nunique()
    return out


def residence_panel(spec, pairs, cohorts=(1914, 1975)):
    from scripts.directories import data_inter
    parts = []
    for y in YEARS:
        f = os.path.join(data_inter, 'simulations', f'{y}_{spec}_origin_residence.parquet')
        if os.path.exists(f):
            parts.append(pd.read_parquet(f))
    if len(parts) < 3 or not any(p.year.iloc[0] == 1992 for p in parts):
        return None
    o = pd.concat(parts); o['comuna'] = o.mother_comuna.astype(int)
    rows = []
    for _, p in pairs.iterrows():
        for col, share in [('comuna_a', p.share_a), ('comuna_b', p.share_b)]:
            s = o[o.comuna == int(p[col])].copy(); s['pair'] = p.pair_id; s['share92'] = share; rows.append(s)
    return pd.concat(rows, ignore_index=True)


def fmt(v, sc=100, nd=1):
    tail = '' if (len(v) < 3 or pd.isna(v[2])) else f' [{sc*v[2]:.{nd}f}]'
    return f'{sc*v[0]:.{nd}f} ({sc*v[1]:.{nd}f}){tail}'


SC = {'pov': (100, 1), 'mapuche': (100, 1), 'schooling': (1, 2)}


def main(spec, threshold, weighted, cohorts=(1914, 1975), ctag='', max_pop=None):
    c = comuna_panel(spec, cohorts)
    pairs = pd.read_parquet(os.path.join(data_final, 'comuna_pairs.parquet'))
    pairs = pairs[{'01sd': pairs.keep_01sd, '5pp': pairs['diff'] >= 0.05, '10pp': pairs['diff'] >= 0.10}[threshold]]
    if max_pop is not None:   # drop pairs involving a comuna above max_pop inhabitants in the 1992 census (residence)
        from scripts.directories import data_inter
        pop = pd.read_parquet(os.path.join(data_inter, 'harmonized', 'census_1992_per.parquet'), columns=['comuna92']).groupby('comuna92').size()
        big = set(pop[pop > max_pop].index.astype(int))
        pairs = pairs[~pairs.comuna_a.isin(big) & ~pairs.comuna_b.isin(big)]
        ctag = ctag + '_nocity'
    d = stack(c, pairs)
    res = {}
    for y in ['pov', 'mapuche', 'schooling']:
        if d[y].notna().sum() > 0 and d.loc[d[y].notna(), 'year'].nunique() >= 3 and 1992 in d.loc[d[y].notna(), 'year'].unique():
            res[y] = pair_did(d, y, weighted)
    tag = f'_{threshold}' + ('_w' if weighted else '') + ctag
    dres = residence_panel(spec, pairs) if not ctag else None
    if dres is not None and len(dres):
        res['pov_res'] = pair_did_residence(dres, 'poor')
    if all(cc in d for cc in CONTROLS) and 'pov' in res:
        res['pov_ctl'] = pair_did(d, 'pov', weighted, CONTROLS)
    tex = ['\\begin{tabular}{lccccc}', '\\toprule', ' & Imputed poverty & Imputed poverty, & Imputed poverty, & Mapuche share & Schooling (years) \\\\',
           ' & & 1992 comuna controls$\\times$year & residence$\\times$year FE & & \\\\', '\\midrule']
    for y_ in YEARS:
        tex.append(f'$\\beta_{{{y_}}}$ & ' + ' & '.join(fmt(res[y][f'b{y_}'], *SC.get(y, (100, 1))) if (y in res and f'b{y_}' in res[y]) else '--' for y in ['pov', 'pov_ctl', 'pov_res', 'mapuche', 'schooling']) + ' \\\\')
    tex.append('\\midrule')
    for y_ in YEARS[1:]:
        tex.append(f'$\\beta_{{{y_}}}-\\beta_{{1992}}$ & ' + ' & '.join(fmt(res[y][f'd{y_}'], *SC.get(y, (100, 1))) if (y in res and f'd{y_}' in res[y]) else '--' for y in ['pov', 'pov_ctl', 'pov_res', 'mapuche', 'schooling']) + ' \\\\')
    r0 = next(iter(res.values()))
    tex += ['\\midrule', f'Pairs & {r0["pairs"]} & & & & \\\\', f'Comunas of origin & {r0["comunas"]} & & & & \\\\', '\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, f'pair_did_{spec}{tag}.tex'), 'w').write('\n'.join(tex) + '\n')
    print('\n'.join(tex))
    OBS = [('floor_dirt', 'Dirt floor'), ('sanit_none', 'No sanitation'), ('elec_none', 'No electricity'), ('water_public', 'Public water'), ('wall_precarious', 'Precarious walls'), ('employed', 'Employed (adults)')]
    res2 = {y: pair_did(d, y, weighted) for y, _ in OBS if y in d and d[y].notna().sum() > 0 and d.loc[d[y].notna(), 'year'].nunique() >= 3}
    if res2:
        t2 = ['\\begin{tabular}{l' + 'c' * len(res2) + '}', '\\toprule', ' & ' + ' & '.join(lab for y, lab in OBS if y in res2) + ' \\\\', '\\midrule']
        for y_ in YEARS:
            t2.append(f'$\\beta_{{{y_}}}$ & ' + ' & '.join(fmt(res2[y][f'b{y_}']) if f'b{y_}' in res2[y] else '--' for y, _ in OBS if y in res2) + ' \\\\')
        t2.append('\\midrule')
        for y_ in YEARS[1:]:
            t2.append(f'$\\beta_{{{y_}}}-\\beta_{{1992}}$ & ' + ' & '.join(fmt(res2[y][f'd{y_}']) if f'd{y_}' in res2[y] else '--' for y, _ in OBS if y in res2) + ' \\\\')
        t2 += ['\\bottomrule', '\\end{tabular}']
        open(os.path.join(results_tables, f'pair_did_observed{tag}.tex'), 'w').write('\n'.join(t2) + '\n'); print('\n'.join(t2))
        if 'floor_dirt' in res2 and threshold == '5pp' and not weighted and not ctag:
            write_macro('didObsFloorDirtBase', 100 * res2['floor_dirt']['b1992'][0])
        for y, r in res2.items():
            for k in ['d2002', 'd2017', 'd2024']:
                if k not in r: continue
                nm = 'didObs' + ''.join(w.capitalize() for w in y.split('_')) + {'d2002': 'TwoThousandTwo', 'd2017': 'Seventeen', 'd2024': 'TwentyFour'}[k] + ('' if threshold == '5pp' and not weighted and not ctag else 'x')
                if nm.endswith('x'): continue
                write_macro(nm, 100 * r[k][0]); write_macro(nm + 'Se', 100 * r[k][1])
    yn = {1992: 'NinetyTwo', 2002: 'TwoThousandTwo', 2017: 'Seventeen', 2024: 'TwentyFour'}
    for y, r in res.items():
        for k, v in r.items():
            if k in ('n', 'pairs', 'comunas'):
                continue
            nm = 'did' + {'pov': 'Pov', 'pov_res': 'PovRes', 'pov_ctl': 'PovCtl', 'mapuche': 'Mp', 'schooling': 'Sch'}[y] + ('Beta' if k[0] == 'b' else 'Diff') + yn[int(k[1:])] + ('' if threshold == '5pp' and not weighted and not ctag else tag.replace('_', '').replace('5pp', '').replace('10pp', 'ten').replace('01sd', 'all').replace('young', 'Young').replace('nocity', 'NoCity'))
            if isinstance(v, (list, tuple)) and len(v) < 2: continue
            write_macro(nm, 100 * v[0] if y != 'schooling' else v[0], '{:.1f}' if y != 'schooling' else '{:.2f}')
            write_macro(nm + 'Se', 100 * v[1] if y != 'schooling' else v[1], '{:.1f}' if y != 'schooling' else '{:.2f}')
    tg = {'5pp': '', '10pp': 'ten', '01sd': 'all'}[threshold]
    write_macro('didPairs' + tg, r0['pairs'], '{:d}'); write_macro('didComunas' + tg, r0['comunas'], '{:d}')
    # figure
    fig, ax = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.6), squeeze=False)
    for j, (y, r) in enumerate(res.items()):
        yrs = [y_ for y_ in YEARS if f'b{y_}' in r]
        b = [r[f'b{y_}'][0] for y_ in yrs]; se = [r[f'b{y_}'][1] for y_ in yrs]
        sc = 100 if y != 'schooling' else 1
        ax[0, j].errorbar(yrs, [sc * x for x in b], yerr=[1.96 * sc * s for s in se], fmt='o-', color='#1f4e79', capsize=3)
        ax[0, j].axvline(1993, color='grey', ls='--', lw=1); ax[0, j].axhline(0, color='grey', lw=0.8)
        ax[0, j].set_title({'pov': 'Imputed poverty (pp per unit share)', 'pov_res': 'Imputed poverty, residence FE', 'pov_ctl': 'Imputed poverty, 1992 controls x year', 'mapuche': 'Mapuche share (pp)', 'schooling': 'Schooling (years)'}[y], fontsize=9)
        ax[0, j].set_xticks(yrs)
    fig.tight_layout(); fig.savefig(os.path.join(results_figures, f'pair_did{tag}.png'), dpi=200)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', default='B'); ap.add_argument('--threshold', default='5pp', choices=['01sd', '5pp', '10pp'])
    ap.add_argument('--weighted', action='store_true'); ap.add_argument('--young', action='store_true'); ap.add_argument('--max-pop', type=int, default=None); a = ap.parse_args()
    main(a.spec, a.threshold, a.weighted, (1950, 1975) if a.young else (1914, 1975), '_young' if a.young else '', a.max_pop)

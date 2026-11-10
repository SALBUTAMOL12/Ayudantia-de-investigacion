#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 exhibits from C_final/groups_wide.parquet:
  Figure: self-identification within closed cohort groups, 2024 vs 1992 (binned scatter),
          and the change by 1992 characteristics.
  Table : (i) micro poverty gap by self-identification at each census (same cohorts),
          (ii) ecological gap: Goodman regression of group poverty on the 1992 indigenous
          share, by year, with population weights; (iii) first-difference (group FE) version.
Standard errors cluster by mother's comuna; simulation uncertainty is carried as the
(weighted) average simulation variance.

Usage: python -m scripts.analysis.part2_selfid_and_gap [--spec B]
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


def wls(y, X, w, cluster):
    X = sm.add_constant(X)
    return sm.WLS(y, X, weights=w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(cluster)[0]})


def binned(g, x, y, w, nb=20):
    edges = np.unique(np.quantile(g[x], np.linspace(0, 1, nb + 1)))
    b = pd.cut(g[x], edges, include_lowest=True)
    return g.groupby(b, observed=True).apply(lambda d: pd.Series({'x': np.average(d[x], weights=d[w]), 'y': np.average(d[y], weights=d[w])}), include_groups=False)


def main(spec):
    g = pd.read_parquet(os.path.join(data_final, 'groups_wide.parquet'))
    YRS = [y for y in [1992, 2002, 2017, 2024] if f'n_{y}' in g]
    g = g[(g.cohort5 >= 1914) & (g.cohort5 <= 1975)].dropna(subset=[f'n_{y}' for y in YRS]).copy()
    for y in YRS:
        g = g[g[f'n_{y}'] >= 20]
    for y in YRS:
        g[f'pi{y}'] = g[f'indigenous_{y}'].fillna(0); g[f'mp{y}'] = g[f'mapuche_{y}'].fillna(0)
    g['w'] = g.n_1992
    g['region_origin'] = (g.mother_comuna // 1000).replace({15: 1, 14: 10, 16: 8}).astype(int)
    # ---------------- Figure: share in t vs t-1, three transitions, all indigenous and Mapuche
    TR = [(a, b) for a, b in zip(YRS[:-1], YRS[1:])] + [(1992, 2024)]
    fig, ax = plt.subplots(1, len(TR), figsize=(4.3 * len(TR), 4.2))
    for j, (t0, t1) in enumerate(TR):
        ax[j].plot([0.005, 1], [0.005, 1], color='grey', lw=1, ls='--')
        b = binned(g, f'pi{t0}', f'pi{t1}', 'w'); ax[j].scatter(b.x, b.y, color='#1f4e79', s=26, label='Any indigenous people')
        bm = binned(g, f'mp{t0}', f'mp{t1}', 'w'); ax[j].scatter(bm.x, bm.y, color='#c0504d', s=26, marker='s', label='Mapuche')
        ax[j].set_xscale('log'); ax[j].set_yscale('log'); ax[j].set_xlim(0.005, 1); ax[j].set_ylim(0.005, 1)
        ax[j].set_xlabel(f'Share in {t0}'); ax[j].set_ylabel(f'Share in {t1}'); ax[j].set_title(f'{t0} to {t1}')
        if j == 0: ax[j].legend(frameon=False, loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(results_figures, 'selfid_transitions.png'), dpi=200)
    # ---------------- Figure: change 1992-2024 by schooling and urban share of the cell in 1992
    g['dpi'] = g.pi2024 - g.pi1992; g['dmp'] = g.mp2024 - g.mp1992
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for j, (col, lab) in enumerate([('schooling_1992', 'Mean adult schooling of the cell in 1992'), ('urban_1992', 'Urban share of the cell in 1992')]):
        q = pd.qcut(g[col], 10, labels=False, duplicates='drop')
        for y, c, nm in [('dpi', '#1f4e79', 'Any indigenous people'), ('dmp', '#c0504d', 'Mapuche')]:
            bb = g.groupby(q).apply(lambda d: pd.Series({'x': np.average(d[col], weights=d.w), 'y': np.average(d[y], weights=d.w)}), include_groups=False)
            ax[j].plot(bb.x, 100 * bb.y, marker='o', color=c, label=nm)
        ax[j].axhline(0, color='grey', lw=1, ls='--'); ax[j].set_xlabel(lab); ax[j].set_ylabel('Change in share 1992-2024 (pp)')
        if j == 0: ax[j].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(results_figures, 'selfid_gradients.png'), dpi=200)
    # ---------------- Table: by region of origin, three years
    RN = {1: "Tarapac\\'a (incl. Arica)", 2: 'Antofagasta', 3: 'Atacama', 4: 'Coquimbo', 5: "Valpara\\'iso", 6: "O'Higgins", 7: 'Maule', 8: "Biob\\'io (incl. \\~Nuble)", 9: "Araucan\\'ia", 10: "Los Lagos (incl. Los R\\'ios)", 11: "Ays\\'en", 12: 'Magallanes', 13: 'Metropolitana'}
    def wavg(sub, col, wcol): return 100 * np.average(sub[col], weights=sub[wcol])
    k = len(YRS)
    rt = ['\\begin{tabular}{lr' + 'c' * (2 * k) + '}', '\\toprule', f' & & \\multicolumn{{{k}}}{{c}}{{Any indigenous people}} & \\multicolumn{{{k}}}{{c}}{{Mapuche}} \\\\',
          f'\\cmidrule(lr){{3-{2 + k}}}\\cmidrule(lr){{{3 + k}-{2 + 2 * k}}}',
          'Region of origin & Persons 1992 & ' + ' & '.join(str(y) for y in YRS) + ' & ' + ' & '.join(str(y) for y in YRS) + ' \\\\', '\\midrule']
    for r_, sub in list(g.groupby('region_origin')) + [('All', g)]:
        rt.append(f"{RN.get(r_, r_)} & {int(sub.n_1992.sum()):,} & " + ' & '.join(f'{wavg(sub, f"pi{y}", f"n_{y}"):.1f}' for y in YRS) + ' & ' +
                  ' & '.join(f'{wavg(sub, f"mp{y}", f"n_{y}"):.1f}' for y in YRS) + ' \\\\')
    rt += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, 'selfid_by_region.tex'), 'w').write('\n'.join(rt) + '\n'); print('\n'.join(rt))
    # ---------------- descriptive regressions of the change on 1992 cell characteristics (macros)
    for y, tag in [('dpi', 'dpi'), ('dmp', 'dmp')]:
        xcol = 'pi1992' if y == 'dpi' else 'mp1992'
        r = wls(g[y], g[['schooling_1992', 'urban_1992', xcol]], g.w, g.mother_comuna)
        write_macro(f'{tag}Schooling', 100 * r.params['schooling_1992'], '{:.2f}'); write_macro(f'{tag}SchoolingSe', 100 * r.bse['schooling_1992'], '{:.2f}')
        write_macro(f'{tag}Urban', 100 * r.params['urban_1992']); write_macro(f'{tag}UrbanSe', 100 * r.bse['urban_1992'])
        write_macro(f'{tag}Pi', 100 * r.params[xcol]); write_macro(f'{tag}PiSe', 100 * r.bse[xcol])
    # persistence slopes without fixed effects (descriptive): share_t on share_{t-1}
    YN = {1992: 'NinetyTwo', 2002: 'TwoThousandTwo', 2017: 'Seventeen', 2024: 'TwentyFour'}
    for t0, t1 in TR:
        nm = YN[t0] + YN[t1]
        r = wls(g[f'pi{t1}'], g[[f'pi{t0}']], g.w, g.mother_comuna); write_macro('persist' + nm, r.params.iloc[1], '{:.2f}')
        r = wls(g[f'mp{t1}'], g[[f'mp{t0}']], g.w, g.mother_comuna); write_macro('persistMp' + nm, r.params.iloc[1], '{:.2f}')
    rm_ = g[g.region_origin == 13]; write_macro('mpRMNinetyTwo', wavg(rm_, 'mp1992', 'n_1992')); write_macro('mpRMTwentyFour', wavg(rm_, 'mp2024', 'n_2024'))
    write_macro('groupsBoth', len(g), '{:d}'); write_macro('groupsPersonsNinetyTwo', g.n_1992.sum() / 1e6, '{:.1f}')
    for y in YRS:
        nm = YN[y]
        write_macro('pi' + nm, wavg(g, f'pi{y}', f'n_{y}')); write_macro('mp' + nm, wavg(g, f'mp{y}', f'n_{y}'))
        write_macro('piSd' + nm, 100 * np.sqrt(np.average((g[f'pi{y}'] - np.average(g[f'pi{y}'], weights=g.w)) ** 2, weights=g.w)))
    print('done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', default='B')
    main(ap.parse_args().spec)

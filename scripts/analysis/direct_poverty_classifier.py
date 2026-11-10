#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix robustness: predict the poverty indicator directly (no income simulation).
Logit and gradient-boosting classifiers on the spec-B shared variables, fitted on CASEN
(weighted), evaluated on held-out comunas (AUC, Brier, calibration), then applied to the census;
the expected headcount of a cell is the mean predicted probability of its members.

Outputs: B_inter/simulations/{year}_D_cells.csv and {year}_D_groups.parquet in the same layout
as script 04 (poor = expected headcount; poor_sd = NaN), plus results/tables/direct_classifier_fit.tex.
Usage: python -m scripts.analysis.direct_poverty_classifier [--year 1992 2017 2024]
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
from scripts.directories import data_inter, results_tables
from scripts.tools.features import design_matrix

HARM = os.path.join(data_inter, 'harmonized'); SIM = os.path.join(data_inter, 'simulations')


def run(year, seed=42):
    d = pd.read_parquet(os.path.join(HARM, f'casen_{year}_hh.parquet'))
    d = d[d.poor_official.notna()].copy()
    X = design_matrix(d, year, 'B'); y = d.poor_official.to_numpy(); w = d.weight.to_numpy()
    g = (d['comuna'] if 'comuna' in d else d['comuna_old']).to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, g))
    models = {'logit': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
              'gbm': HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)}
    rows, probs = [], {}
    for name, m in models.items():
        kw = {'logisticregression__sample_weight': w[tr]} if name == 'logit' else {'sample_weight': w[tr]}
        m.fit(X.iloc[tr], y[tr], **kw)
        p = m.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y[te], p, sample_weight=w[te]); brier = brier_score_loss(y[te], p, sample_weight=w[te])
        cal = np.average(p, weights=w[te]) - np.average(y[te], weights=w[te])
        rows.append(dict(year=year, model=name, auc=auc, brier=brier, calib_gap=cal))
        kw = {'logisticregression__sample_weight': w} if name == 'logit' else {'sample_weight': w}
        m.fit(X, y, **kw); probs[name] = m
        print(f'[{year} {name}] held-out AUC {auc:.3f} Brier {brier:.3f} mean p - mean y {cal:+.3f}')
    # census
    hh = pd.read_parquet(os.path.join(HARM, f'census_{year}_hh.parquet')).reset_index(drop=True)
    hh['region13'] = (hh.comuna // 1000).replace({15: 1, 14: 10, 16: 8}).astype(float)
    Xc = design_matrix(hh, year, 'B')[list(X.columns)]
    per = pd.read_parquet(os.path.join(HARM, f'census_{year}_per.parquet'))
    pos = pd.Series(np.arange(len(hh)), index=hh.hh_id); per = per[per.hh_id.isin(pos.index)].copy()
    pidx = pos.reindex(per.hh_id).to_numpy()
    out_cells, out_groups = [], []
    ind = per.indigenous.fillna(2).astype(int).to_numpy(); urb = per.urban.fillna(0).astype(int).to_numpy()
    by = per.birth_year.to_numpy(); coh = ((by >= 1914) & (by <= 1978) & per.mother_comuna.notna().to_numpy()).astype(int)
    cells = {'eth_x_urban': ind * 2 + urb, 'eth_x_cohort': ind * 2 + coh, 'all': np.zeros(len(per), int)}
    gkey = per[['cohort5', 'female', 'mother_comuna']]; gvalid = gkey.notna().all(axis=1).to_numpy()
    gcodes, guniq = pd.factorize(pd.MultiIndex.from_frame(gkey[gvalid]))
    for name, m in probs.items():
        p_hh = m.predict_proba(Xc)[:, 1]; pp = p_hh[pidx]
        for cname, c in cells.items():
            n = np.bincount(c); s = np.bincount(c, weights=pp)
            for k in np.flatnonzero(n):
                out_cells.append(dict(year=year, spec='D_' + name, cell=cname, code=int(k), n=n[k], poor=s[k] / n[k], poor_sd=np.nan))
        gn = np.bincount(gcodes); gs = np.bincount(gcodes, weights=pp[gvalid])
        out_groups.append(pd.DataFrame({'cohort5': guniq.get_level_values(0), 'female': guniq.get_level_values(1),
                                        'mother_comuna': guniq.get_level_values(2), 'n': gn, 'poor': gs / gn, 'poor_sd': np.nan,
                                        'year': year, 'spec': 'D_' + name}))
        print(f'[{year} {name}] census expected headcount {np.mean(pp):.4f}')
    pd.DataFrame(out_cells).to_csv(os.path.join(SIM, f'{year}_D_cells.csv'), index=False)
    pd.concat(out_groups).to_parquet(os.path.join(SIM, f'{year}_D_groups.parquet'), index=False)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--year', type=int, nargs='*', default=[1992, 2017, 2024])
    a = ap.parse_args()
    rows = []
    for y in a.year:
        rows += run(y)
    m = pd.DataFrame(rows)
    f = os.path.join(results_tables, 'direct_classifier_fit.csv')
    if os.path.exists(f):
        old = pd.read_csv(f); m = pd.concat([old[~old.year.isin(m.year)], m])
    m.to_csv(f, index=False)
    t = ['\\begin{tabular}{llccc}', '\\toprule', 'Year & Classifier & AUC & Brier & Mean $\\hat p$ $-$ mean $y$ \\\\', '\\midrule']
    for _, r in m.sort_values(['year', 'model']).iterrows():
        t.append(f'{int(r.year)} & {"Logit" if r.model == "logit" else "Gradient boosting"} & {r.auc:.3f} & {r.brier:.3f} & {100*r.calib_gap:+.1f} \\\\')
    t += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(results_tables, 'direct_classifier_fit.tex'), 'w').write('\n'.join(t) + '\n')
    print('\n'.join(t))

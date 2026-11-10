#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 04: simulate household income in the census with the first-stage models and
aggregate poverty indicators by ethnicity (Part 1) and by closed cohort group (Part 2).

For each year and spec, R replications:
  spec A : beta_r ~ N(beta_hat, V_cluster); u = eta_c + eps, both normal (ELL).
  spec B/K : beta_r from a comuna-cluster bootstrap refit of the LASSO (fixed alpha);
             u = eta_c (normal) + eps drawn from the empirical residuals of the fitted-value
             decile of the household (heteroskedastic, non-normal).
Poverty: y_hh = exp(lny_pc) * n ; poor if y_hh < line_hh (official line of the year).

Outputs in B_inter/simulations/:
  {year}_{spec}_cells.csv      mean and sd over replications of the poverty rate, FGT1,
                               extreme-poverty rate by person-level cell
  {year}_{spec}_groups.parquet cohort5 x sex x mother's comuna groups: n, poverty mean/sd
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from scripts.directories import data_inter
from scripts.tools.features import design_matrix

HARM = os.path.join(data_inter, 'harmonized')
MOD = os.path.join(data_inter, 'models')
SIM = os.path.join(data_inter, 'simulations')
os.makedirs(SIM, exist_ok=True)
LP2024 = {'rent': 352359.0, 'own': 237094.0}
LI2024 = {'rent': 234906.0, 'own': 158063.0}


def census_lines(hh, year):
    n = hh.n_people.to_numpy(float)
    if year == 2024:
        rent = hh.tenure_rented.fillna(0).to_numpy() == 1
        lp = np.where(rent, LP2024['rent'], LP2024['own']); li = np.where(rent, LI2024['rent'], LI2024['own'])
        return lp * n ** 0.7, li * n ** 0.7
    if year == 2017:
        # the 2017 census has no tenure question: impute P(renter | x) with a logit fitted on CASEN 2017 and
        # draw renter status in each replication (see run()); return both lines and the probability.
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        c = pd.read_parquet(os.path.join(HARM, 'casen_2017_hh.parquet'))
        lr = c.groupby('renter')[['lp_pc', 'li_pc']].first()
        Xc = design_matrix(c, 2017, 'B'); cols = list(Xc.columns)
        lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xc, c.renter.to_numpy(), logisticregression__sample_weight=c.weight.to_numpy())
        p_rent = lg.predict_proba(design_matrix(hh, 2017, 'B')[cols])[:, 1]
        print(f'   renter status imputed: mean P(rent) {p_rent.mean():.3f} (CASEN weighted share {np.average(c.renter, weights=c.weight):.3f})')
        lines = {'own': (lr.loc[0.0, 'lp_pc'] * n ** 0.7, lr.loc[0.0, 'li_pc'] * n ** 0.7),
                 'rent': (lr.loc[1.0, 'lp_pc'] * n ** 0.7, lr.loc[1.0, 'li_pc'] * n ** 0.7)}
        return lines, p_rent
    if year == 2002:
        # urban status is an imputed probability in the 2002 census; lines drawn per replication (Agostini/MIDEPLAN 2003 lines)
        p_urb = hh.urban.to_numpy(float)
        return {'own': (29473.0 * n, 16842.0 * n), 'rent': (43712.0 * n, 21856.0 * n)}, p_urb
    if year == 1992:
        urb = hh.urban.to_numpy() == 1
        lp = np.where(urb, 25750.0, 17362.0); li = np.where(urb, 12875.0, 9924.0)
        return lp * n, li * n
    raise ValueError(year)


def bootstrap_lasso(year, spec, R, seed):
    """Comuna-cluster bootstrap refits of the LASSO with the CV-chosen alpha. Cached."""
    f = os.path.join(MOD, f'{year}_{spec}_boot.npz')
    if os.path.exists(f):
        z = np.load(f)
        if z['coef'].shape[0] >= R:
            return z['coef'][:R], z['intercept'][:R]
    m = joblib.load(os.path.join(MOD, f'{year}_{spec}.joblib'))
    d = pd.read_parquet(os.path.join(HARM, f'casen_{year}_hh.parquet'))
    d = d[(d.ypc > 0) & d.ypc.notna()].copy()
    X = design_matrix(d, year, spec)[m['columns']].to_numpy(np.float64)
    y = np.log(d.ypc.to_numpy()); w = d.weight.to_numpy()
    cl = d['comuna'] if 'comuna' in d else d['comuna_old']
    codes, uniq = pd.factorize(cl)
    idx_by_c = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    coefs, ints = [], []
    for r in range(R):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_c[k] for k in pick])
        pipe = make_pipeline(StandardScaler(), Lasso(alpha=m['alpha'], max_iter=3000, precompute=False))
        pipe.fit(X[idx], y[idx], lasso__sample_weight=w[idx])
        sc, la = pipe.named_steps['standardscaler'], pipe.named_steps['lasso']
        coefs.append(la.coef_ / sc.scale_); ints.append(la.intercept_ - np.sum(la.coef_ * sc.mean_ / sc.scale_))
        if r % 50 == 0:
            print(f'   bootstrap {r}/{R}')
    coefs, ints = np.array(coefs), np.array(ints)
    np.savez(f, coef=coefs, intercept=ints)
    return coefs, ints


def run(year, spec, R=200, seed=7):
    rng = np.random.default_rng(seed)
    m = joblib.load(os.path.join(MOD, f'{year}_{spec}.joblib'))
    print(f'[{year} {spec}] loading census ...')
    hh = pd.read_parquet(os.path.join(HARM, f'census_{year}_hh.parquet'))
    hh['region13'] = (hh.comuna // 1000).replace({15: 1, 14: 10, 16: 8}).astype(float)
    per = pd.read_parquet(os.path.join(HARM, f'census_{year}_per.parquet'))
    # align persons to households
    hh = hh.reset_index(drop=True)
    pos = pd.Series(np.arange(len(hh)), index=hh.hh_id)
    per = per[per.hh_id.isin(pos.index)].copy()
    pidx = pos.reindex(per.hh_id).to_numpy()
    n = hh.n_people.to_numpy(float)
    cl = census_lines(hh, year)
    p_rent = None
    if isinstance(cl[0], dict):
        lines_by, p_rent = cl
    else:
        line_hh, line_ext = cl
    X = design_matrix(hh, year, spec)[m['columns']].to_numpy(np.float32)
    # ---- coefficient draws
    if spec == 'A':
        bc = m['beta_cov']
        B = rng.multivariate_normal(bc['params'], bc['cov'], size=R)
        ints, coefs = B[:, 0], B[:, 1:]
    else:
        coefs, ints = bootstrap_lasso(year, spec, R, seed)
    # cluster effects by comuna
    ccodes, cuniq = pd.factorize(hh.comuna)
    sig_eta, sig_eps = m['sig_eta'], m['sig_eps']
    # decile assignment for empirical residuals (spec B/K)
    yhat0 = m['estimator'].predict(design_matrix(hh, year, spec)[m['columns']])
    dec = np.clip(np.searchsorted(m['dec_edges'][1:-1], yhat0, side='right'), 0, 9)
    res_pool = {k: v for k, v in m['resid_by_dec'].items()}
    # ---- person-level cells for Part 1
    ind = per.indigenous.fillna(2).astype(int).to_numpy()       # 0 non, 1 indig, 2 missing
    urb = per.urban.fillna(0).astype(int).to_numpy()
    pueblo = per.pueblo.astype(int).to_numpy(); pueblo = np.where(pueblo < 0, 99, pueblo)
    reg = per.region.astype(int).to_numpy()
    by = per.birth_year.to_numpy()
    coh = ((by >= 1914) & (by <= 1978) & per.mother_comuna.notna().to_numpy()).astype(int)
    mp = per.mapuche.fillna(2).astype(int).to_numpy()          # 0 non-Mapuche (incl. other peoples), 1 Mapuche, 2 missing
    cells = {
        'eth_x_urban': ind * 2 + urb,                         # 0..5
        'eth_x_cohort': ind * 2 + coh,                        # odd codes = closed-cohort sample
        'mp_x_urban': mp * 2 + urb,
        'mp_x_cohort': mp * 2 + coh,
        'pueblo': pueblo,                                     # 0 none, 1..12, 99 missing
        'region_x_eth': reg * 10 + np.minimum(ind, 2),
        'head_eth_x_urban': (hh.head_indigenous.fillna(2).astype(int).to_numpy()[pidx]) * 2 + urb,
        'all': np.zeros(len(per), int),
    }
    # ---- groups for Part 2 (closed cohorts): cohort5 x female x mother's comuna
    gkey = per[['cohort5', 'female', 'mother_comuna']].copy()
    gvalid = gkey.notna().all(axis=1).to_numpy()
    gcodes, guniq = pd.factorize(pd.MultiIndex.from_frame(gkey[gvalid]))
    gfull = np.full(len(per), -1); gfull[gvalid] = gcodes
    # accumulators
    acc = {}
    for name, c in cells.items():
        K = c.max() + 1
        acc[name] = {'n': np.bincount(c, minlength=K).astype(float),
                     'poor': np.zeros((R, K)), 'fgt1': np.zeros((R, K)), 'ext': np.zeros((R, K))}
    G = len(guniq)
    gn = np.bincount(gcodes, minlength=G).astype(float)
    gpoor = np.zeros((R, G)); gfgt = np.zeros((R, G))
    # origin comuna x residence comuna cells for the closed-cohort sample (residence-by-year FE version of Part 2)
    okey = per[['mother_comuna', 'comuna92']].copy(); okey['coh'] = coh
    ovalid = (okey.coh == 1).to_numpy() & okey[['mother_comuna', 'comuna92']].notna().all(axis=1).to_numpy()
    ocodes, ouniq = pd.factorize(pd.MultiIndex.from_frame(okey.loc[ovalid, ['mother_comuna', 'comuna92']]))
    O = len(ouniq); on = np.bincount(ocodes, minlength=O).astype(float); opoor = np.zeros((R, O))
    # cells x schooling bracket (0-8, 9-12, 13+) for the predicted-probability (proxy) estimator
    sb = pd.cut(per.schooling, [-1, 8, 12, 99], labels=False).astype(float)
    skey = per[['cohort5', 'female', 'mother_comuna']].copy(); skey['sb'] = sb
    svalid = skey.notna().all(axis=1).to_numpy() & (coh == 1)
    scodes, suniq = pd.factorize(pd.MultiIndex.from_frame(skey[svalid]))
    S_ = len(suniq); sn = np.bincount(scodes, minlength=S_).astype(float); spoor = np.zeros((R, S_))
    print(f'[{year} {spec}] simulating R={R} on {len(hh):,} households / {len(per):,} persons ...')
    for r in range(R):
        lny = X @ coefs[r].astype(np.float32) + ints[r]
        eta = rng.normal(0, sig_eta, len(cuniq))[ccodes]
        if spec == 'A':
            eps = rng.normal(0, sig_eps, len(hh))
        else:
            eps = np.empty(len(hh), np.float32)
            for k in range(10):
                mk = dec == k
                pool = res_pool.get(k, np.concatenate(list(res_pool.values())))
                eps[mk] = rng.choice(pool, mk.sum(), replace=True)
        y_hh = np.exp(lny + eta + eps) * n
        if p_rent is not None:
            rent_r = rng.random(len(hh)) < p_rent
            line_hh = np.where(rent_r, lines_by['rent'][0], lines_by['own'][0]); line_ext = np.where(rent_r, lines_by['rent'][1], lines_by['own'][1])
        poor = (y_hh < line_hh).astype(float)
        ext = (y_hh < line_ext).astype(float)
        gap = np.clip(1 - y_hh / line_hh, 0, None)
        pp, pe, pg = poor[pidx], ext[pidx], gap[pidx]
        for name, c in cells.items():
            K = acc[name]['n'].shape[0]
            acc[name]['poor'][r] = np.bincount(c, weights=pp, minlength=K)
            acc[name]['ext'][r] = np.bincount(c, weights=pe, minlength=K)
            acc[name]['fgt1'][r] = np.bincount(c, weights=pg, minlength=K)
        gpoor[r] = np.bincount(gcodes, weights=pp[gvalid], minlength=G)
        opoor[r] = np.bincount(ocodes, weights=pp[ovalid], minlength=O)
        spoor[r] = np.bincount(scodes, weights=pp[svalid], minlength=S_)
        gfgt[r] = np.bincount(gcodes, weights=pg[gvalid], minlength=G)
        if r % 25 == 0:
            print(f'   rep {r}: national poverty {pp.mean():.4f}')
    rows = []
    for name, a in acc.items():
        nz = a['n'] > 0
        for k in np.flatnonzero(nz):
            rows.append(dict(year=year, spec=spec, cell=name, code=int(k), n=a['n'][k],
                             poor=a['poor'][:, k].mean() / a['n'][k], poor_sd=(a['poor'][:, k] / a['n'][k]).std(ddof=1),
                             fgt1=a['fgt1'][:, k].mean() / a['n'][k], fgt1_sd=(a['fgt1'][:, k] / a['n'][k]).std(ddof=1),
                             extreme=a['ext'][:, k].mean() / a['n'][k], extreme_sd=(a['ext'][:, k] / a['n'][k]).std(ddof=1)))
    pd.DataFrame(rows).to_csv(os.path.join(SIM, f'{year}_{spec}_cells.csv'), index=False)
    gdf = pd.DataFrame({'cohort5': guniq.get_level_values(0), 'female': guniq.get_level_values(1),
                        'mother_comuna': guniq.get_level_values(2), 'n': gn,
                        'poor': gpoor.mean(0) / gn, 'poor_sd': (gpoor / gn).std(0, ddof=1),
                        'fgt1': gfgt.mean(0) / gn, 'fgt1_sd': (gfgt / gn).std(0, ddof=1)})
    gdf['year'] = year; gdf['spec'] = spec
    gdf.to_parquet(os.path.join(SIM, f'{year}_{spec}_groups.parquet'), index=False)
    odf = pd.DataFrame({'mother_comuna': ouniq.get_level_values(0), 'comuna92': ouniq.get_level_values(1), 'n': on,
                        'poor': opoor.mean(0) / on, 'poor_sd': (opoor / on).std(0, ddof=1), 'year': year, 'spec': spec})
    odf.to_parquet(os.path.join(SIM, f'{year}_{spec}_origin_residence.parquet'), index=False)
    sdf = pd.DataFrame({'cohort5': suniq.get_level_values(0), 'female': suniq.get_level_values(1), 'mother_comuna': suniq.get_level_values(2),
                        'sb': suniq.get_level_values(3), 'n': sn, 'poor': spoor.mean(0) / sn, 'poor_sd': (spoor / sn).std(0, ddof=1), 'year': year, 'spec': spec})
    sdf.to_parquet(os.path.join(SIM, f'{year}_{spec}_cells_sch.parquet'), index=False)
    print(pd.DataFrame(rows).query("cell=='eth_x_urban' or cell=='all'").to_string())


# =============================================================================
# Validation: same simulation on the CASEN sample, compared with official poverty
# =============================================================================
def validate(year, spec, R=50, seed=11):
    rng = np.random.default_rng(seed)
    m = joblib.load(os.path.join(MOD, f'{year}_{spec}.joblib'))
    d = pd.read_parquet(os.path.join(HARM, f'casen_{year}_hh.parquet'))
    d = d[(d.ypc > 0) & d.ypc.notna()].reset_index(drop=True)
    X = design_matrix(d, year, spec)[m['columns']].to_numpy(np.float32)
    n = d.numper.to_numpy(float); w = d.weight.to_numpy() * n     # person weights
    line = d.line_hh.to_numpy()
    cl = d['comuna'] if 'comuna' in d else d['comuna_old']
    ccodes, cuniq = pd.factorize(cl)
    if spec == 'A':
        bc = m['beta_cov']; B = rng.multivariate_normal(bc['params'], bc['cov'], size=R); ints, coefs = B[:, 0], B[:, 1:]
    else:
        coefs, ints = bootstrap_lasso(year, spec, R, seed)
    yhat0 = m['estimator'].predict(design_matrix(d, year, spec)[m['columns']])
    dec = np.clip(np.searchsorted(m['dec_edges'][1:-1], yhat0, side='right'), 0, 9)
    sims = []
    for r in range(R):
        lny = X @ coefs[r].astype(np.float32) + ints[r]
        eta = rng.normal(0, m['sig_eta'], len(cuniq))[ccodes]
        if spec == 'A':
            eps = rng.normal(0, m['sig_eps'], len(d))
        else:
            eps = np.empty(len(d), np.float32)
            for k in range(10):
                mk = dec == k; eps[mk] = rng.choice(m['resid_by_dec'][k], mk.sum(), replace=True)
        y_hh = np.exp(lny + eta + eps) * n
        sims.append(np.average(y_hh < line, weights=w))
    ok = d.poor_official.notna().to_numpy(); off = np.average(d.poor_official[ok], weights=w[ok])
    print(f'[validate {year} {spec}] official person poverty {off:.4f} | simulated {np.mean(sims):.4f} (sd {np.std(sims):.4f}) | '
          f'mean fitted-only {np.average(np.exp(yhat0) * n < line, weights=w):.4f}')
    vf = os.path.join(MOD, 'validation.csv')
    row = pd.DataFrame([dict(year=year, spec=spec, official=off, simulated=np.mean(sims), sim_sd=np.std(sims))])
    if os.path.exists(vf):
        old = pd.read_csv(vf); old = old[~((old.year == year) & (old.spec == spec))]; row = pd.concat([old, row])
    row.to_csv(vf, index=False)
    return off, np.mean(sims)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, nargs='*', default=[2024, 1992])
    ap.add_argument('--spec', nargs='*', default=['A', 'B'])
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--validate', action='store_true')
    a = ap.parse_args()
    for y in a.year:
        for s in a.spec:
            if a.validate:
                validate(y, s)
            else:
                run(y, s, a.R)



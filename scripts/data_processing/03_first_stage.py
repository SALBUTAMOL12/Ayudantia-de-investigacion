#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 03: first-stage income models on CASEN, by year and specification.

  A : OLS (ELL / Agostini-style) on the Agostini variable list -> normal cluster + idiosyncratic errors
  B : LASSO (alpha by 10-fold comuna-grouped CV) on everything shared with the census
  K : LASSO on the core set common to 1992/2002/2024
  G : HistGradientBoosting on spec-B variables (R² benchmark only, appendix)

Dependent variable: log per-capita household income (ln ypc), weighted by expr.
Out-of-sample R² from a 20% hold-out of comunas (grouped split) and from grouped CV.

Outputs (B_inter/models/{year}_{spec}.joblib): fitted estimator, column list, residual
information (sigma_eta, sigma_eps, residuals by fitted-value decile) and metrics in
B_inter/models/metrics.csv (appended).
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from scripts.directories import data_inter
from scripts.tools.features import design_matrix

HARM = os.path.join(data_inter, 'harmonized')
MOD = os.path.join(data_inter, 'models')
os.makedirs(MOD, exist_ok=True)


def load(year):
    d = pd.read_parquet(os.path.join(HARM, f'casen_{year}_hh.parquet'))
    d = d[(d.ypc > 0) & d.ypc.notna()].copy()
    d['lny'] = np.log(d.ypc)
    d['cluster'] = d['comuna'] if 'comuna' in d else d['comuna_old']
    return d


def residual_components(resid, cluster, w):
    """ELL decomposition u = eta_c + eps: variance of cluster means (bias-corrected) and
    within variance."""
    df = pd.DataFrame({'r': resid, 'c': cluster, 'w': w})
    g = df.groupby('c')
    means = g.apply(lambda x: np.average(x.r, weights=x.w), include_groups=False)
    n = g.size()
    within = df.r - df.c.map(means)
    sig_eps2 = np.average(within ** 2, weights=df.w) * (len(df) / (len(df) - len(means)))
    sig_eta2 = max(np.var(means, ddof=1) - sig_eps2 * np.mean(1.0 / n), 0.0)
    return float(np.sqrt(sig_eta2)), float(np.sqrt(sig_eps2))


def fit(year, spec, seed=42):
    d = load(year)
    X = design_matrix(d, year, spec)
    y = d.lny.to_numpy(); w = d.weight.to_numpy(); g = d.cluster.to_numpy()
    # hold-out 20% of comunas
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, g))
    cols = list(X.columns)
    if spec == 'A':
        est = LinearRegression()
    elif spec in ('B', 'K'):
        cv = list(GroupKFold(n_splits=10).split(X.iloc[tr], y[tr], g[tr]))
        est = make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-4, 0, 30), cv=cv, n_jobs=-1, max_iter=5000, random_state=seed, precompute=False))
    elif spec == 'G':
        est = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
    fitkw = {'sample_weight': w[tr]}
    if spec in ('B', 'K'):
        fitkw = {'lassocv__sample_weight': w[tr]}
    est.fit(X.iloc[tr], y[tr], **fitkw)
    r2_tr = r2_score(y[tr], est.predict(X.iloc[tr]), sample_weight=w[tr])
    r2_te = r2_score(y[te], est.predict(X.iloc[te]), sample_weight=w[te])
    # refit on full sample for prediction, keep hold-out metrics
    if spec in ('B', 'K'):
        alpha = est.named_steps['lassocv'].alpha_
        from sklearn.linear_model import Lasso
        est_full = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000, precompute=False))
        est_full.fit(X, y, lasso__sample_weight=w)
        nnz = int((est_full.named_steps['lasso'].coef_ != 0).sum())
    else:
        est_full = est.__class__(**est.get_params()); est_full.fit(X, y, sample_weight=w); alpha = np.nan; nnz = len(cols)
    yhat = est_full.predict(X)
    resid = y - yhat
    sig_eta, sig_eps = residual_components(resid, g, w)
    # residuals by decile of fitted value (for empirical, heteroskedastic draws)
    dec = pd.qcut(yhat, 10, labels=False, duplicates='drop')
    resid_by_dec = {int(k): resid[dec == k].astype('float32') for k in np.unique(dec)}
    dec_edges = np.quantile(yhat, np.linspace(0, 1, 11))
    # OLS covariance for spec A (HC-robust, cluster by comuna)
    beta_cov = None
    if spec == 'A':
        import statsmodels.api as sm
        Xc = sm.add_constant(X.to_numpy())
        res = sm.WLS(y, Xc, weights=w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(g)[0]})
        beta_cov = {'params': res.params, 'cov': res.cov_params(), 'names': ['const'] + cols}
    out = dict(year=year, spec=spec, estimator=est_full, columns=cols, sig_eta=sig_eta, sig_eps=sig_eps,
               resid_by_dec=resid_by_dec, dec_edges=dec_edges, beta_cov=beta_cov,
               r2_train=r2_tr, r2_test=r2_te, alpha=alpha, n_nonzero=nnz, n_obs=len(d))
    joblib.dump(out, os.path.join(MOD, f'{year}_{spec}.joblib'))
    m = pd.DataFrame([{k: out[k] for k in ['year', 'spec', 'r2_train', 'r2_test', 'alpha', 'n_nonzero', 'n_obs', 'sig_eta', 'sig_eps']}])
    mf = os.path.join(MOD, 'metrics.csv')
    if os.path.exists(mf):
        old = pd.read_csv(mf); old = old[~((old.year == year) & (old.spec == spec))]
        m = pd.concat([old, m])
    m.to_csv(mf, index=False)
    print(f'[{year} {spec}] n={len(d)} cols={len(cols)} R2 train={r2_tr:.3f} test(held-out comunas)={r2_te:.3f} '
          f'alpha={alpha} nnz={nnz} sig_eta={sig_eta:.3f} sig_eps={sig_eps:.3f}')
    if spec in ('B', 'K'):
        coef = pd.Series(est_full.named_steps['lasso'].coef_, index=cols)
        print(coef[coef != 0].abs().sort_values(ascending=False).head(15).to_string())
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, nargs='*', default=[2024, 1992])
    ap.add_argument('--spec', nargs='*', default=['A', 'B', 'K', 'G'])
    a = ap.parse_args()
    for y in a.year:
        for s in a.spec:
            fit(y, s)

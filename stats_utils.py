"""
stats_utils.py
==============
Small statistical helpers shared across the figure scripts:
- quadratic (income^2 / pct_black^2) curvature tests
- incremental R^2 with city fixed effects
- local (kernel-weighted) variance for the GWR instability correction
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree


def quadratic_fit(df: pd.DataFrame, x: str, y: str):
    """Fit y ~ x + x^2 (OLS). Returns (fitted statsmodels result, is_inverted_u).

    is_inverted_u is True iff the quadratic term is negative AND
    significant at p < 0.05 (matches the paper's "checkmark = inverted-U"
    convention in Figures 5 & 6).
    """
    d = df[[x, y]].dropna().copy()
    d["_x2"] = d[x] ** 2
    model = smf.ols(f"{y} ~ {x} + _x2", data=d).fit()
    quad_coef = model.params["_x2"]
    quad_p = model.pvalues["_x2"]
    is_inverted_u = bool((quad_coef < 0) and (quad_p < 0.05))
    return model, is_inverted_u, quad_p


def incremental_r2(df: pd.DataFrame, outcome: str, predictors: list[str],
                    city_col: str = "city") -> tuple[float, float, float]:
    """R^2 of `outcome ~ predictors + C(city)` minus R^2 of `outcome ~ C(city)`.

    Returns (incremental_r2, full_r2, base_r2).
    """
    cols = [outcome, city_col] + predictors
    d = df[cols].dropna().copy()
    base = smf.ols(f"{outcome} ~ C({city_col})", data=d).fit()
    rhs = " + ".join(predictors + [f"C({city_col})"])
    full = smf.ols(f"{outcome} ~ {rhs}", data=d).fit()
    return full.rsquared - base.rsquared, full.rsquared, base.rsquared


def fit_coefs(df: pd.DataFrame, outcome: str, predictors: list[str],
              extra_terms: list[str] | None = None):
    """OLS of outcome ~ predictors (+ extra_terms), no city FE (used per-city
    in Figure 9, and for the pooled Figure 7 right panel where city FE IS
    included via extra_terms=['C(city)']).

    Returns dict: {predictor: (coef, se, pvalue)}.
    """
    extra_terms = extra_terms or []
    cols = [outcome] + predictors + (["city"] if "C(city)" in extra_terms else [])
    d = df[cols].dropna().copy() if "city" in cols else df[[outcome] + predictors].dropna().copy()
    rhs = " + ".join(predictors + extra_terms)
    model = smf.ols(f"{outcome} ~ {rhs}", data=d).fit()
    out = {}
    for p in predictors:
        out[p] = (model.params.get(p, np.nan), model.bse.get(p, np.nan), model.pvalues.get(p, np.nan))
    return out, model


def bisquare_adaptive_weights(coords: np.ndarray, bw: int):
    """For each point i, find its `bw` nearest neighbors (including itself)
    and compute adaptive-bisquare kernel weights:
        w_ij = (1 - (d_ij / d_max_i)^2)^2   for d_ij < d_max_i, else 0
    where d_max_i is the distance to the bw-th nearest neighbor of i.

    Returns (neighbor_idx, weights) as lists of arrays, one per point --
    this mirrors what an adaptive-bisquare GWR kernel uses internally, and
    is the basis for the Section 2.8 local-variance instability diagnostic.
    """
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=min(bw, len(coords)))
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    dmax = dists[:, -1].reshape(-1, 1)
    dmax[dmax == 0] = 1e-12
    w = (1 - (dists / dmax) ** 2) ** 2
    w[dists >= dmax] = 0.0
    return idxs, w


def local_weighted_variance(values: np.ndarray, neighbor_idx: np.ndarray, weights: np.ndarray):
    """Weighted variance of `values` within each point's kernel neighborhood."""
    v = values[neighbor_idx]  # (n, k)
    wsum = weights.sum(axis=1)
    wsum[wsum == 0] = np.nan
    wmean = (weights * v).sum(axis=1) / wsum
    wvar = (weights * (v - wmean.reshape(-1, 1)) ** 2).sum(axis=1) / wsum
    return wvar

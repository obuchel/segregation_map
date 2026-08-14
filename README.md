# Reproducing Figures 5–9

This pipeline reproduces the five new figures described in
`paper_sections_updated_with_figures.docx` (Sections 3.X–3.W) directly
from the `features_<City>_diversity.json` files you uploaded — no other
inputs are needed.

## Run it

```bash
pip install pandas numpy scipy matplotlib statsmodels mgwr geopandas pyproj --break-system-packages
python3 run_all.py
```

Figure 8 (the GWR maps) is the slow step — it runs a golden-section
AICc bandwidth search plus a full GWR fit for each of the five cities.
Everything else runs in a few seconds.

## Files

| File | Produces |
|---|---|
| `data_loader.py` | Loads the 5 `*_diversity.json` files into one tidy per-tract table |
| `stats_utils.py` | Quadratic-curvature test, incremental R², kernel-weighted local variance |
| `fig5_income_diversity.py` | Figure 5 |
| `fig6_ethnicity_diversity.py` | Figure 6 |
| `fig7_r2_comparison.py` | Figure 7 |
| `fig8_gwr_maps.py` | Figure 8 (GWR) |
| `fig9_percity_coefficients.py` | Figure 9 |
| `run_all.py` | Runs everything, in order |

## Modeling choices the source data doesn't spell out

The JSON gives raw tract-level fields; the write-up describes the
*outcome* of several analytical choices without fully specifying them.
Where that happened, I made an explicit, documented choice rather than
guess silently — flagging them here so you can override any of them
before this goes in the paper:

- **pct_black / pct_other_race / pct_white**: `pct_white` is
  `white_alone / total`, not a residual — `hispanic_or_latino` is its
  own separate field in the source data and is deliberately excluded
  from all three composition predictors rather than folded into
  `pct_white` or `pct_other_race`. So `pct_black + pct_other_race +
  pct_white` no longer sums to 1 in tracts with a `hispanic_or_latino`
  population; that share just isn't represented among the predictors
  used in the regressions/GWR. (`pct_white` itself isn't used as a
  predictor in any of Figures 5–9 — only `pct_black`, `pct_other_race`,
  and income are — so this only changes the stored column's definition,
  not any figure.)
- **Per-tract diversity score**: each tract has 4 entropy values
  (mobility × {in, out}, mentions × {in, out}). Figures 5, 6, 7, 9 each
  need one number per tract per network, so I averaged in/out per
  network. If the original analysis used only `out` (or only `in`),
  swap the `np.nanmean([...])` calls in `data_loader.py` for a single
  field.
- **Mobility degree** (Figure 7's third outcome): same in/out averaging.
- **Income deciles / pct_black deciles**: computed within-city via
  `pd.qcut`, independent of the existing `Income_Quantile` field (which
  the docx's own terminology note says is six-category, not a decile).
- **Ethnicity predictor (Figures 7, 8, 9)**: all three now use
  `ethnic_diversity` — a normalized Shannon entropy across all 8 Census
  race/ethnicity categories per tract (0 = single group, 1 = evenly
  spread across all 8) — in place of separate `pct_black`/`pct_other_race`
  shares. The two-share version left `pct_white` as an implicit omitted
  reference category in every regression (with city fixed effects, the
  intercept absorbs whatever share `pct_black + pct_other_race` didn't
  cover); a single entropy score has no reference category at all.
  Figure 6 still bins tracts by `pct_black` decile directly, since that
  figure is a descriptive plot (not a regression with an omitted
  category) — let me know if you'd like it switched too for consistency.
  The GWR model is `mentions_diversity ~ Income_10k + ethnic_diversity`;
  Figure 7/9's regressions are `mentions_diversity ~ ethnic_diversity
  [+ Income_10k + Income_10k_sq] [+ C(city)]`.
- **GWR mechanics**: adaptive bisquare kernel, bandwidth (# nearest
  neighbors) selected per city by golden-section search minimizing AICc
  (`mgwr.sel_bw.Sel_BW`), exactly as Section 2.7 describes. Coordinates
  are reprojected from lon/lat to the local UTM zone in meters so
  distances are metrically correct.
- **Local-variance instability correction** (Section 2.8): implemented
  from scratch (mgwr doesn't expose this) as a kernel-weighted variance
  of `ethnic_diversity` within each tract's own bandwidth neighborhood,
  using the same bisquare kernel and bandwidth as that city's fitted GWR
  model. Bottom quartile → flagged "unstable" → excluded from the
  corrected row.
- **County boundaries in Figure 8**: the source caption mentions "real
  county-boundary context (gray lines)." This sandbox has no network
  path to Census TIGER/Line shapefiles, so Figure 8 here is tract
  centroids only. If you have a county shapefile/GeoJSON for these five
  metros, set `COUNTY_SHAPEFILE` at the top of `fig8_gwr_maps.py` to its
  path and add a couple of lines to plot it under the scatter (geopandas
  is already installed, e.g. `gpd.read_file(path).boundary.plot(ax=ax, color='gray', linewidth=0.5)`).

Because of these choices, the exact percentages you'll see printed in
the terminal (e.g. "% significant" per city) won't necessarily match
the specific numbers quoted in Section 2.8's prose — the analytical
structure is the same, but a couple of unstated parameters (in/out
averaging, exact income scaling) can shift results by a few points.
Worth a side-by-side check before these regenerated figures replace
the original ones in the paper.

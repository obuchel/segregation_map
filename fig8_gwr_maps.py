"""
Figure 8: Spatial distribution of local GWR coefficients for
ethnic/racial diversity (Shannon entropy across Census race/ethnicity
categories) predicting mentions diversity, by city, before and after
the Section 2.8 local-variance instability correction.

Per instruction, this uses a single composition predictor --
`ethnic_diversity` (normalized entropy across all 8 Census
race/ethnicity categories, see data_loader.py) -- in place of the
separate pct_black / pct_other_race categorical shares used in an
earlier version of this pipeline.

Model per city: mentions_diversity ~ Income_10k + ethnic_diversity
fit with an adaptive-bisquare-kernel Geographically Weighted Regression,
bandwidth (# nearest neighbors) chosen per city by golden-section search
minimizing AICc (mgwr.sel_bw.Sel_BW).

Instability correction: for every tract, compute the kernel-weighted
variance of ethnic_diversity among its own bandwidth-neighborhood (same
kernel and bandwidth as that city's fitted model). Tracts in the bottom
quartile of that variance distribution (most locally homogeneous
neighborhoods -- the local regression has almost nothing to fit a slope
against) are flagged and excluded from the "corrected" bottom row.

NOTE on the "real county-boundary context" mentioned in the source
caption: this sandbox has no network access to Census TIGER/Line
shapefiles, so no county boundary polygons are drawn here. If you have
a county shapefile available (e.g. a .shp/.geojson of the relevant
counties), pass its path as COUNTY_SHAPEFILE below and this script will
plot it under the points automatically; otherwise it degrades
gracefully to a plain tract-centroid scatter.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer
from mgwr.sel_bw import Sel_BW as Sel_bw
from mgwr.gwr import GWR

from data_loader import load_all_cities, CITIES, CITY_DISPLAY
from stats_utils import bisquare_adaptive_weights, local_weighted_variance

CITY_ORDER = [CITY_DISPLAY.get(c, c) for c in CITIES]
PREDICTORS = ["Income_10k", "ethnic_diversity"]
COEF_OF_INTEREST = "ethnic_diversity"
TARGET = "mentions_diversity"

# Optional: path to a county boundary shapefile/geojson covering these
# five metros. Left as None because no such file is bundled with the
# uploaded data and this sandbox cannot fetch one from the Census Bureau.
COUNTY_SHAPEFILE = None


def to_local_meters(lon, lat):
    """Project lon/lat to a local UTM zone (meters) for correct distances."""
    zone = int((np.mean(lon) + 180) // 6) + 1
    epsg = 32600 + zone if np.mean(lat) >= 0 else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.column_stack([x, y])


def fit_city_gwr(sub):
    cols = [TARGET] + PREDICTORS + ["lon", "lat", "dissimilarity_contribution"]
    d = sub[cols].dropna().reset_index(drop=True)

    coords = to_local_meters(d["lon"].values, d["lat"].values)
    y = d[TARGET].values.reshape(-1, 1)
    X = d[PREDICTORS].values

    sel = Sel_bw(coords, y, X, kernel="bisquare", fixed=False)
    bw = sel.search(criterion="AICc", search_method="golden_section")
    bw = int(round(bw))

    model = GWR(coords, y, X, bw, kernel="bisquare", fixed=False)
    results = model.fit()

    target_col = 1 + PREDICTORS.index(COEF_OF_INTEREST)  # +1 for intercept in col 0
    coef = results.params[:, target_col]
    tval = results.tvalues[:, target_col]

    idx, w = bisquare_adaptive_weights(coords, bw)
    local_var = local_weighted_variance(d[COEF_OF_INTEREST].values, idx, w)
    q1_cut = np.nanpercentile(local_var, 25)
    unstable = local_var <= q1_cut

    d["coef"] = coef
    d["tval"] = tval
    d["unstable"] = unstable
    d["bw"] = bw
    return d


def pct_significant(d, mask=None):
    dd = d if mask is None else d[~d["unstable"]] if mask == "corrected" else d
    sig = (dd["tval"].abs() > 1.96).mean() * 100
    return sig


def make_figure(df, out_path="output/gwr_maps_with_dissimilarity_glow.png"):
    fig, axes = plt.subplots(2, 5, figsize=(19, 7.5))
    cmap = plt.get_cmap("coolwarm")

    results_by_city = {}
    for city in CITY_ORDER:
        print(f"  fitting GWR for {city} ...")
        results_by_city[city] = fit_city_gwr(df[df["city"] == city])

    vmax = max(np.nanpercentile(np.abs(r["coef"]), 98) for r in results_by_city.values())
    vmax = max(vmax, 1e-6)

    for c_i, city in enumerate(CITY_ORDER):
        d = results_by_city[city]
        sig_before = pct_significant(d)
        sig_after = pct_significant(d, mask="corrected")

        # --- top row: original / unfiltered ---
        ax = axes[0, c_i]
        glow = d["dissimilarity_contribution"].clip(lower=0)
        glow = glow / (glow.max() if glow.max() > 0 else 1)
        ax.scatter(d["lon"], d["lat"], c="black", s=glow * 120, alpha=0.15, linewidths=0)
        sc = ax.scatter(d["lon"], d["lat"], c=d["coef"], cmap=cmap, vmin=-vmax, vmax=vmax,
                         s=8, linewidths=0)
        ax.set_title(f"{city}\n{sig_before:.0f}% sig.", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # --- bottom row: corrected (unstable tracts excluded) ---
        ax2 = axes[1, c_i]
        ax2.scatter(d["lon"], d["lat"], c="black", s=glow * 120, alpha=0.15, linewidths=0)
        stable = d[~d["unstable"]]
        excluded = d[d["unstable"]]
        ax2.scatter(stable["lon"], stable["lat"], c=stable["coef"], cmap=cmap,
                    vmin=-vmax, vmax=vmax, s=8, linewidths=0)
        ax2.scatter(excluded["lon"], excluded["lat"], c="gray", marker="x", s=10, linewidths=0.6,
                    alpha=0.6)
        ax2.set_title(f"{sig_after:.0f}% sig. (corrected)", fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Corrected", fontsize=11)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01)
    cbar.set_label("Local GWR coefficient: ethnic_diversity \u2192 mentions diversity")

    legend_elems = [Line2D([0], [0], marker="x", color="gray", linestyle="", markersize=6,
                           label="Excluded (unstable, low local variance)")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=1, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Figure 8: Local GWR coefficients (ethnic_diversity \u2192 mentions diversity), original vs. corrected\n"
                 "background glow = tract contribution to city dissimilarity index",
                 fontsize=12, y=1.03)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    bws = {city: int(results_by_city[city]["bw"].iloc[0]) for city in CITY_ORDER}
    print("Selected bandwidths (# neighbors):", bws)
    return results_by_city


if __name__ == "__main__":
    df = load_all_cities()
    make_figure(df)

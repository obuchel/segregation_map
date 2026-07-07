"""
gwr_spatial_analysis.py

Runs Geographically Weighted Regression (GWR) per city to test whether the
association between racial composition and connection diversity (controlling
for income) varies across space within a city - rather than assuming one
fixed coefficient applies uniformly, as an ordinary regression with city
fixed effects would.

For each city, fits:
    outcome ~ Income_k + Income_k_sq + pct_black + pct_hispanic + pct_other_race

with coefficients allowed to vary smoothly by location (adaptive bisquare
kernel, bandwidth selected automatically per city via golden section search
on AICc). Extracts the LOCAL coefficient and t-value for whichever variable
you're interested in (defaults to pct_black) at every tract, and maps them.

Inputs expected in --input-dir:
    features_<City>_diversity.json   (Income, ethnicity, Centroid [lon, lat])
    node_diversity_combined.csv       (from compute_node_diversity.py)

Outputs (in --output-dir):
    gwr_dataset.csv                 (merged dataset used, with coordinates)
    gwr_local_coefficients.csv      (per-tract local coefficient + t-value)
    gwr_spatial_maps.png            (the map itself)

Usage:
    python gwr_spatial_analysis.py --input-dir . --output-dir . \\
        --outcome mentions_in_entropy --variable pct_black

Requires: mgwr, libpysal
    pip install mgwr libpysal --break-system-packages
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

CITIES = ["Chicago", "Dallas", "Detroit", "NewYork", "Philadelphia"]

# Predictor order matters - must match how columns are indexed after fitting.
# Index 0 is always the intercept that GWR adds automatically.
PREDICTORS = ["Income_k", "Income_k_sq", "pct_black", "pct_hispanic", "pct_other_race"]


def build_dataset(input_dir):
    rows = []
    for city in CITIES:
        path = os.path.join(input_dir, f"features_{city}_diversity.json")
        with open(path, "r") as f:
            data = json.load(f)
        for feat in data["features"]:
            props = feat["properties"]
            eth = props.get("ethnicity")
            income = props.get("Income")
            centroid = props.get("Centroid")
            if eth is None or eth.get("total", 0) == 0 or income is None or centroid is None:
                continue
            total = eth["total"]
            hispanic = eth.get("hispanic_or_latino", 0) or 0
            white = eth.get("white_alone", 0) or 0
            black = eth.get("black_alone", 0) or 0
            other_race = total - hispanic - white - black
            rows.append({
                "city": city, "GEOID": str(props["GEOID"]).zfill(11),
                "lon": centroid[0], "lat": centroid[1],
                "Income": income,
                "pct_black": black / total,
                "pct_hispanic": hispanic / total,
                "pct_other_race": other_race / total,
            })
    geo_df = pd.DataFrame(rows)

    div_path = os.path.join(input_dir, "node_diversity_combined.csv")
    div_df = pd.read_csv(div_path, dtype={"GEOID": str})
    div_df["GEOID"] = div_df["GEOID"].str.zfill(11)

    merged = geo_df.merge(div_df, on=["city", "GEOID"], how="inner")
    merged["Income_k"] = merged["Income"] / 1000
    merged["Income_k_sq"] = merged["Income_k"] ** 2
    return merged


def run_gwr_per_city(df, outcome, variable):
    var_idx = 1 + PREDICTORS.index(variable)  # +1 for the intercept column GWR prepends
    all_results = []

    for city in CITIES:
        sub = df[df.city == city].dropna(subset=[outcome]).copy().reset_index(drop=True)
        if len(sub) < 30:
            print(f"  {city}: skipped (only {len(sub)} tracts with data)")
            continue
        print(f"{city}: {len(sub)} tracts...")

        coords = list(zip(sub["lon"], sub["lat"]))
        y = sub[outcome].values.reshape(-1, 1)
        X = sub[PREDICTORS].values

        t0 = time.time()
        bw = Sel_BW(coords, y, X).search()
        results = GWR(coords, y, X, bw).fit()
        print(f"  bandwidth={bw:.0f}, R2={results.R2:.3f}, took {time.time()-t0:.1f}s")

        sub["local_coef"] = results.params[:, var_idx]
        sub["local_t"] = results.tvalues[:, var_idx]
        sub["local_sig"] = np.abs(sub["local_t"]) > 1.96
        all_results.append(sub[["city", "GEOID", "lon", "lat", variable,
                                 "local_coef", "local_t", "local_sig"]])

    return pd.concat(all_results, ignore_index=True)


def plot_maps(gwr_results, variable, outcome, output_path):
    cities_present = gwr_results["city"].unique()
    n = len(cities_present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    vmax = gwr_results["local_coef"].abs().quantile(0.98)

    for ax, city in zip(axes, cities_present):
        sub = gwr_results[gwr_results.city == city]
        sig = sub[sub.local_sig]
        nonsig = sub[~sub.local_sig]

        ax.scatter(nonsig["lon"], nonsig["lat"], c=nonsig["local_coef"],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=8, alpha=0.25, edgecolors="none")
        sc = ax.scatter(sig["lon"], sig["lat"], c=sig["local_coef"],
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=14, alpha=0.95,
                        edgecolors="k", linewidths=0.2)

        pct_sig = (sub["local_sig"].sum() / len(sub)) * 100
        ax.set_title(f"{city}\n({pct_sig:.0f}% tracts significant)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.05, aspect=40)
    # CONFIRMED (not assumed) via direct RGB check of RdBu_r: negative values
    # render blue, positive values render red. Get this backwards and every
    # reader draws the opposite conclusion from the map.
    cbar.set_label(f"Local GWR coefficient: {variable} \u2192 {outcome} (blue=negative, red=positive)")

    plt.suptitle(f"Where does {variable} independently predict {outcome}? (GWR, controlling for other predictors)",
                 fontsize=13, y=1.05)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--outcome", default="mentions_in_entropy",
                         help="Column from node_diversity_combined.csv to use as the outcome")
    parser.add_argument("--variable", default="pct_black", choices=PREDICTORS,
                         help="Which predictor's local coefficient to map")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building dataset...")
    df = build_dataset(args.input_dir)
    dataset_path = os.path.join(args.output_dir, "gwr_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Wrote {dataset_path} ({len(df)} tracts)\n")

    print(f"Running GWR per city (outcome={args.outcome}, variable={args.variable})...")
    gwr_results = run_gwr_per_city(df, args.outcome, args.variable)
    results_path = os.path.join(args.output_dir, "gwr_local_coefficients.csv")
    gwr_results.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")

    print("\nSummary by city:")
    for city in gwr_results["city"].unique():
        sub = gwr_results[gwr_results.city == city]
        print(f"  {city}: mean={sub['local_coef'].mean():.3f}, "
              f"% significant={sub['local_sig'].mean()*100:.0f}%, "
              f"% positive-and-sig={((sub['local_coef']>0)&sub['local_sig']).mean()*100:.0f}%, "
              f"% negative-and-sig={((sub['local_coef']<0)&sub['local_sig']).mean()*100:.0f}%")

    plot_path = os.path.join(args.output_dir, "gwr_spatial_maps.png")
    plot_maps(gwr_results, args.variable, args.outcome, plot_path)
    print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()

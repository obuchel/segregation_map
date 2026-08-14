"""
run_all.py
==========
Reproduces Figures 5-9 from "Mapping Virtual Segregation by Spatial
Patterns" (Data & Methodology / Results additions) from the raw
features_<City>_diversity.json files.

Usage:
    python3 run_all.py

Expects features_<City>_diversity.json for City in
{Chicago, Detroit, Philadelphia, Dallas, NewYork} in the same directory
(or pass a different folder via data_dir).

Outputs (written to ./output/):
    figure5_per_city_income_diversity.png
    figure6_ethnicity_diversity_regression.png
    figure7_joint_income_ethnicity_regression.png
    figure9_per_city_mentions_coefficients.png
    gwr_maps_with_dissimilarity_glow.png     (Figure 8; takes the
                                               longest -- one GWR
                                               bandwidth search + fit
                                               per city)
"""
import time
from pathlib import Path

from data_loader import load_all_cities
import fig5_income_diversity as fig5
import fig6_ethnicity_diversity as fig6
import fig7_r2_comparison as fig7
import fig9_percity_coefficients as fig9
import fig8_gwr_maps as fig8


def main():
    Path("output").mkdir(exist_ok=True)

    print("Loading and preparing tract-level data for all 5 cities...")
    df = load_all_cities()
    print(f"  {len(df)} tracts loaded.\n")

    t0 = time.time()
    print("Figure 5 (income x network diversity, per city)...")
    fig5.make_figure(df)

    print("Figure 6 (pct_black x network diversity, pooled)...")
    fig6.make_figure(df)

    print("Figure 7 (incremental R^2 / joint coefficients)...")
    fig7.make_figure(df)

    print("Figure 9 (per-city ethnicity coefficients)...")
    fig9.make_figure(df)

    print("Figure 8 (GWR spatial maps -- this is the slow one)...")
    fig8.make_figure(df)

    print(f"\nDone in {time.time() - t0:.1f}s. See ./output/")


if __name__ == "__main__":
    main()

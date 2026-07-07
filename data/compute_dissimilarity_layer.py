"""
compute_dissimilarity_layer.py

Adds a spatial "dissimilarity" layer to your existing features_<City>_diversity.json
files: for each census tract, computes its individual contribution to the
city's Black/non-Black dissimilarity index (the standard segregation measure).

The city-wide dissimilarity index D is:
    D = 0.5 * sum_i |black_i/total_black_city - nonblack_i/total_nonblack_city|

Each tract's own term inside that sum (before the 0.5x and summation) tells
you how much that specific tract pulls the city away from an even
distribution - large values mean the tract is disproportionately Black OR
disproportionately non-Black relative to the city as a whole. This lets you
see WHERE within a city segregation is being driven from, not just the
single city-wide number.

Adds to each tract's properties:
    dissimilarity_contribution   (the per-tract term, sign preserved:
                                   positive = disproportionately Black,
                                   negative = disproportionately non-Black)
    city_dissimilarity_index     (same single value repeated on every tract
                                   in that city, for convenience/display)

Usage:
    python compute_dissimilarity_layer.py --input-dir . --output-dir .
"""

import argparse
import json
import os
import shutil

CITIES = ["Chicago", "Dallas", "Detroit", "NewYork", "Philadelphia"]


def compute_for_city(input_dir, output_dir, city):
    in_path = os.path.join(input_dir, f"features_{city}_diversity.json")
    with open(in_path, "r") as f:
        data = json.load(f)

    feats_with_eth = [f for f in data["features"] if f["properties"].get("ethnicity")]
    total_black = sum(f["properties"]["ethnicity"].get("black_alone", 0) or 0 for f in feats_with_eth)
    total_all = sum(f["properties"]["ethnicity"]["total"] for f in feats_with_eth)
    total_nonblack = total_all - total_black

    if total_black == 0 or total_nonblack == 0:
        print(f"{city}: cannot compute (zero Black or non-Black population), skipping")
        return

    d_index = 0.0
    contributions = {}
    for f in feats_with_eth:
        geoid = str(f["properties"]["GEOID"]).zfill(11)
        black_i = f["properties"]["ethnicity"].get("black_alone", 0) or 0
        nonblack_i = f["properties"]["ethnicity"]["total"] - black_i
        # Signed contribution: positive = tract has an excess SHARE of the
        # city's Black population relative to its share of non-Black
        # population (i.e. disproportionately Black); negative = the reverse.
        signed_contribution = (black_i / total_black) - (nonblack_i / total_nonblack)
        contributions[geoid] = signed_contribution
        d_index += abs(signed_contribution)
    d_index *= 0.5

    missing = 0
    for feat in data["features"]:
        geoid = str(feat["properties"]["GEOID"]).zfill(11)
        contribution = contributions.get(geoid)
        if contribution is None:
            missing += 1
        feat["properties"]["dissimilarity_contribution"] = contribution
        feat["properties"]["city_dissimilarity_index"] = round(d_index, 4)

    out_path = os.path.join(output_dir, f"features_{city}_diversity.json")
    if os.path.abspath(out_path) == os.path.abspath(in_path):
        backup_path = in_path + ".dissimilarity.bak"
        if not os.path.exists(backup_path):
            shutil.copy2(in_path, backup_path)

    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"{city}: D = {d_index:.3f}, wrote {out_path} ({missing} tracts had no ethnicity data)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    for city in CITIES:
        compute_for_city(args.input_dir, output_dir, city)


if __name__ == "__main__":
    main()

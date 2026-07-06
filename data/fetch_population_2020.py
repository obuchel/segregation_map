"""
fetch_population_2020.py

Fetches total population per census tract from the 2020 DECENNIAL Census
(not ACS - this is the actual once-a-decade headcount, exact counts rather
than survey estimates) and adds it to your features_<City>_diversity.json
files as a "Population" property, matching the POP_FIELD constant already
in PublicationMapPage.jsx. No code changes needed in the component - just
run this once and re-deploy the enriched files.

Dataset: dec/pl (2020 Census PL 94-171 Redistricting Data)
Variable: P1_001N = Total population

Usage:
    export CENSUS_API_KEY="your_key_here"
    python fetch_population_2020.py --input-dir . --output-dir .

By default this OVERWRITES the same features_<City>_diversity.json files
in place (after backing them up with a .bak extension), so the app doesn't
need any new filenames. Use --output-dir to write elsewhere instead.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import requests

CENSUS_API_BASE = "https://api.census.gov/data"
DATASET = "dec/pl"          # 2020 Decennial Census, PL 94-171 Redistricting Data
POP_VARIABLE = "P1_001N"    # Total population
YEAR = 2020

CITIES = ["Chicago", "Dallas", "Detroit", "NewYork", "Philadelphia"]


def parse_geoid(geoid):
    geoid = str(geoid).zfill(11)
    return geoid[0:2], geoid[2:5], geoid[5:11]


def group_by_state_county(geoids):
    groups = defaultdict(list)
    for g in geoids:
        state, county, _ = parse_geoid(g)
        groups[(state, county)].append(g)
    return groups


def fetch_county_population(state, county, api_key, session, max_retries=3):
    url = f"{CENSUS_API_BASE}/{YEAR}/{DATASET}"
    params = {
        "get": POP_VARIABLE,
        "for": "tract:*",
        "in": f"state:{state} county:{county}",
        "key": api_key,
    }
    for attempt in range(1, max_retries + 1):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            rows = resp.json()
            header, records = rows[0], rows[1:]
            return header, records
        elif resp.status_code == 204:
            return None, []
        else:
            print(f"  [warn] state={state} county={county} attempt {attempt} "
                  f"-> HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            time.sleep(1.5 * attempt)
    return None, []


def build_population_lookup(all_geoids, api_key):
    groups = group_by_state_county(all_geoids)
    lookup = {}
    session = requests.Session()

    print(f"Fetching 2020 Census population for {len(groups)} county groups...")
    for i, ((state, county), _) in enumerate(groups.items(), start=1):
        print(f"  [{i}/{len(groups)}] state={state} county={county}")
        header, records = fetch_county_population(state, county, api_key, session)
        if header is None:
            continue

        pop_i = header.index(POP_VARIABLE)
        state_i = header.index("state")
        county_i = header.index("county")
        tract_i = header.index("tract")

        for rec in records:
            geoid = rec[state_i] + rec[county_i] + rec[tract_i]
            try:
                lookup[geoid] = int(rec[pop_i])
            except (TypeError, ValueError):
                lookup[geoid] = None

        time.sleep(0.2)

    return lookup


def enrich_city(input_dir, output_dir, city, lookup):
    in_path = os.path.join(input_dir, f"features_{city}_diversity.json")
    with open(in_path, "r") as f:
        data = json.load(f)

    missing = 0
    for feat in data["features"]:
        geoid = str(feat["properties"]["GEOID"]).zfill(11)
        pop = lookup.get(geoid)
        if pop is None:
            missing += 1
        feat["properties"]["Population"] = pop

    out_path = os.path.join(output_dir, f"features_{city}_diversity.json")
    if os.path.abspath(out_path) == os.path.abspath(in_path):
        backup_path = in_path + ".bak"
        shutil.copy2(in_path, backup_path)
        print(f"  backed up original to {backup_path}")

    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"{city}: wrote {out_path} ({missing} tracts had no population match)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".",
                         help="Directory containing features_<City>_diversity.json")
    parser.add_argument("--output-dir", default=None,
                         help="Where to write updated files. Defaults to --input-dir "
                              "(overwrites in place, after backing up with .bak)")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("ERROR: export CENSUS_API_KEY='your_key_here' first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    all_geoids = []
    for city in CITIES:
        path = os.path.join(args.input_dir, f"features_{city}_diversity.json")
        with open(path, "r") as f:
            data = json.load(f)
        all_geoids += [feat["properties"]["GEOID"] for feat in data["features"]]

    print(f"Loaded {len(all_geoids)} tract GEOIDs across {len(CITIES)} cities.")
    lookup = build_population_lookup(all_geoids, api_key)
    print(f"Fetched population for {len(lookup)} tracts.")

    for city in CITIES:
        enrich_city(args.input_dir, output_dir, city, lookup)


if __name__ == "__main__":
    main()

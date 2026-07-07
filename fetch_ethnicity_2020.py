"""
fetch_ethnicity_2020.py

Fetches race/ethnicity per census tract from the 2020 DECENNIAL Census
(table P2: "Hispanic or Latino, and Not Hispanic or Latino by Race" - same
dataset as fetch_population_2020.py's population table, just a different
table). These are exact counts from the actual headcount, not ACS survey
estimates.

Adds an "ethnicity" property to each tract in your existing
features_<City>_diversity.json files with counts for:
    total, hispanic_or_latino,
    white_alone, black_alone, native_american_alone, asian_alone,
    pacific_islander_alone, some_other_race_alone, two_or_more_races
(all "non-Hispanic" except hispanic_or_latino, per the P2 table's own
definitions - i.e. this is a standard mutually-exclusive race x ethnicity
breakdown, not double-counted).

Usage:
    export CENSUS_API_KEY="your_key_here"
    python fetch_ethnicity_2020.py --input-dir . --output-dir .

By default OVERWRITES the same features_<City>_diversity.json files in
place (backing up originals with .bak first, unless one already exists
from a previous run - e.g. from fetch_population_2020.py).
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
DATASET = "dec/pl"   # 2020 Census PL 94-171 Redistricting Data (same as population)
YEAR = 2020

# P2 table: Hispanic or Latino, and Not Hispanic or Latino by Race.
# All race categories here are "Not Hispanic or Latino" except the
# hispanic_or_latino variable itself - this is the standard mutually
# exclusive race x ethnicity split used in segregation research.
VARIABLES = {
    "P2_001N": "total",
    "P2_002N": "hispanic_or_latino",
    "P2_005N": "white_alone",
    "P2_006N": "black_alone",
    "P2_007N": "native_american_alone",
    "P2_008N": "asian_alone",
    "P2_009N": "pacific_islander_alone",
    "P2_010N": "some_other_race_alone",
    "P2_011N": "two_or_more_races",
}

CITIES = ["Chicago", "Dallas", "Detroit", "NewYork", "Philadelphia"]


def parse_geoid(geoid):
    geoid = str(geoid).zfill(11)
    return geoid[0:2], geoid[2:5]


def group_by_state_county(geoids):
    groups = defaultdict(list)
    for g in geoids:
        state, county = parse_geoid(g)
        groups[(state, county)].append(g)
    return groups


def fetch_county_ethnicity(state, county, api_key, session, max_retries=3):
    url = f"{CENSUS_API_BASE}/{YEAR}/{DATASET}"
    params = {
        "get": ",".join(VARIABLES.keys()),
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


def build_ethnicity_lookup(all_geoids, api_key):
    groups = group_by_state_county(all_geoids)
    lookup = {}
    session = requests.Session()

    print(f"Fetching 2020 Census race/ethnicity for {len(groups)} county groups...")
    for i, ((state, county), expected_geoids) in enumerate(groups.items(), start=1):
        print(f"  [{i}/{len(groups)}] state={state} county={county} (expect {len(expected_geoids)} tracts)")
        header, records = fetch_county_ethnicity(state, county, api_key, session)
        if header is None:
            print(f"    [MISS] state={state} county={county}: API returned no data at all "
                  f"({len(expected_geoids)} tracts expected)")
            continue

        idx = {code: header.index(code) for code in VARIABLES}
        state_i = header.index("state")
        county_i = header.index("county")
        tract_i = header.index("tract")

        returned_geoids = set()
        for rec in records:
            geoid = rec[state_i] + rec[county_i] + rec[tract_i]
            returned_geoids.add(geoid)
            entry = {}
            for code, name in VARIABLES.items():
                try:
                    entry[name] = int(rec[idx[code]])
                except (TypeError, ValueError):
                    entry[name] = None
            lookup[geoid] = entry

        # Flag counties where the API returned noticeably fewer tracts than
        # your own feature files say should exist there - this is how we
        # catch silent partial failures (no HTTP error, just fewer rows).
        expected_set = {str(g).zfill(11) for g in expected_geoids}
        shortfall = expected_set - returned_geoids
        if shortfall:
            print(f"    [PARTIAL] state={state} county={county}: API returned "
                  f"{len(returned_geoids)} tracts, but {len(shortfall)} expected tracts "
                  f"are missing from the response. Sample missing GEOIDs: "
                  f"{list(shortfall)[:5]}")

        time.sleep(0.2)

    return lookup


def enrich_city(input_dir, output_dir, city, lookup):
    in_path = os.path.join(input_dir, f"features_{city}_diversity.json")
    with open(in_path, "r") as f:
        data = json.load(f)

    missing = 0
    for feat in data["features"]:
        geoid = str(feat["properties"]["GEOID"]).zfill(11)
        ethnicity = lookup.get(geoid)
        if ethnicity is None:
            missing += 1
        feat["properties"]["ethnicity"] = ethnicity

    out_path = os.path.join(output_dir, f"features_{city}_diversity.json")
    if os.path.abspath(out_path) == os.path.abspath(in_path):
        backup_path = in_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(in_path, backup_path)
            print(f"  backed up original to {backup_path}")

    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"{city}: wrote {out_path} ({missing} tracts had no ethnicity match)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".",
                         help="Directory containing features_<City>_diversity.json")
    parser.add_argument("--output-dir", default=None,
                         help="Where to write updated files. Defaults to --input-dir")
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
    lookup = build_ethnicity_lookup(all_geoids, api_key)
    print(f"Fetched ethnicity data for {len(lookup)} tracts.")

    for city in CITIES:
        enrich_city(args.input_dir, output_dir, city, lookup)


if __name__ == "__main__":
    main()

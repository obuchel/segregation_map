"""
fill_missing_via_crosswalk.py

Your features_<City>_diversity.json files use 2010-vintage census tract
GEOIDs (consistent with the underlying mobility/mentions data being from
2013). The Census API's 2020 datasets only recognize 2020-vintage tract
GEOIDs. Between the 2010 and 2020 census, ~11,000-23,000 tracts nationally
were split, merged, or renumbered - so a meaningful fraction of your GEOIDs
(5-28% depending on city, per your last run) simply don't exist in the
2020 API at all under their old ID.

This script uses the Census Bureau's official 2020-to-2010 Census Tract
Relationship File to find which 2020 tract(s) each "missing" 2010 tract
became, fetches population + ethnicity data for those 2020 tract(s), and
backfills your existing features_<City>_diversity.json files - without
touching tracts that already matched successfully.

Source of the relationship file (downloaded automatically if not present
locally): https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/tab20_tract20_tract10_natl.txt
(~50-100MB, all US tract-to-tract intersections, pipe-delimited, per
Census's own documented layout - see
https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2020-comp-record-layout.html)

Usage:
    export CENSUS_API_KEY="your_key_here"
    python fill_missing_via_crosswalk.py --input-dir . --output-dir .

Approximation note: when a 2010 tract was split into multiple 2020 tracts,
this sums population/ethnicity counts across all 2020 tracts that overlap
it by more than MIN_AREA_SHARE of the 2010 tract's land area (default 10%),
to avoid pulling in unrelated population from tiny boundary-adjustment
slivers. This is a standard approximation for tract-vintage crosswalks,
not an exact reallocation - note this in your methods/limitations section.
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
DATASET = "dec/pl"
YEAR = 2020
RELATIONSHIP_FILE_URL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/tab20_tract20_tract10_natl.txt"
MIN_AREA_SHARE = 0.10  # ignore 2020 tracts overlapping <10% of the 2010 tract's land area

CITIES = ["Chicago", "Dallas", "Detroit", "NewYork", "Philadelphia"]

# Population (P1) + ethnicity (P2) variables, fetched together in one pass
POP_VARIABLE = "P1_001N"
ETHNICITY_VARIABLES = {
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
ALL_VARIABLES = [POP_VARIABLE] + list(ETHNICITY_VARIABLES.keys())


def download_relationship_file(local_path):
    if os.path.exists(local_path):
        print(f"Using existing relationship file: {local_path}")
        return
    print(f"Downloading relationship file from {RELATIONSHIP_FILE_URL} ...")
    print("(This is a large national file - may take a few minutes.)")
    resp = requests.get(RELATIONSHIP_FILE_URL, stream=True, timeout=120)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"Downloaded to {local_path}")


def build_crosswalk(local_path, needed_2010_geoids):
    """Returns {geoid_2010: [(geoid_2020, arealand_part), ...]}, sorted by
    descending arealand_part, filtered to needed_2010_geoids only (to keep
    memory reasonable - the national file covers ~85,000 2020 tracts)."""
    needed = set(needed_2010_geoids)
    crosswalk = defaultdict(list)

    with open(local_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        # Detect whether the file has a header row matching documented column names
        cols = first_line.strip().split("|")
        has_header = "GEOID_TRACT_10" in cols
        if has_header:
            col_idx = {name: i for i, name in enumerate(cols)}
        else:
            # Documented column order (no header row variant)
            col_names = [
                "OID_TRACT_20", "GEOID_TRACT_20", "NAMELSAD_TRACT_20",
                "AREALAND_TRACT_20", "AREAWATER_TRACT_20", "MTFCC_TRACT_20", "FUNCSTAT_TRACT_20",
                "OID_TRACT_10", "GEOID_TRACT_10", "NAMELSAD_TRACT_10",
                "AREALAND_TRACT_10", "AREAWATER_TRACT_10", "MTFCC_TRACT_10", "FUNCSTAT_TRACT_10",
                "AREALAND_PART", "AREAWATER_PART",
            ]
            col_idx = {name: i for i, name in enumerate(col_names)}
            f.seek(0)  # no header consumed, reprocess first line as data

        geoid20_i = col_idx["GEOID_TRACT_20"]
        geoid10_i = col_idx["GEOID_TRACT_10"]
        arealand10_i = col_idx["AREALAND_TRACT_10"]
        arealand_part_i = col_idx["AREALAND_PART"]

        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= max(geoid20_i, geoid10_i, arealand10_i, arealand_part_i):
                continue
            geoid10 = parts[geoid10_i].strip()
            if geoid10 not in needed:
                continue
            geoid20 = parts[geoid20_i].strip()
            try:
                area10 = float(parts[arealand10_i])
                area_part = float(parts[arealand_part_i])
            except ValueError:
                continue
            share = (area_part / area10) if area10 > 0 else 0
            if share >= MIN_AREA_SHARE:
                crosswalk[geoid10].append((geoid20, share))

    for geoid10 in crosswalk:
        crosswalk[geoid10].sort(key=lambda x: -x[1])

    return crosswalk


def fetch_2020_tract_data(geoid20_list, api_key):
    """Fetches population + ethnicity for a specific list of 2020 tract
    GEOIDs, grouped by county to minimize API calls."""
    by_county = defaultdict(list)
    for g in geoid20_list:
        by_county[(g[0:2], g[2:5])].append(g)

    result = {}
    session = requests.Session()
    for (state, county), geoids_in_county in by_county.items():
        url = f"{CENSUS_API_BASE}/{YEAR}/{DATASET}"
        params = {
            "get": ",".join(ALL_VARIABLES),
            "for": "tract:*",
            "in": f"state:{state} county:{county}",
            "key": api_key,
        }
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  [warn] fetch failed for state={state} county={county}: "
                  f"HTTP {resp.status_code}", file=sys.stderr)
            continue
        rows = resp.json()
        header, records = rows[0], rows[1:]
        idx = {v: header.index(v) for v in ALL_VARIABLES}
        state_i, county_i, tract_i = header.index("state"), header.index("county"), header.index("tract")
        wanted = set(geoids_in_county)
        for rec in records:
            geoid = rec[state_i] + rec[county_i] + rec[tract_i]
            if geoid not in wanted:
                continue
            result[geoid] = {v: int(rec[idx[v]]) for v in ALL_VARIABLES}
        time.sleep(0.2)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--relationship-file", default="tab20_tract20_tract10_natl.txt",
                         help="Local path to save/reuse the relationship file")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("ERROR: export CENSUS_API_KEY='your_key_here' first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Find which GEOIDs are actually missing population/ethnicity data
    # across your existing enriched files
    city_data = {}
    missing_by_city = {}
    all_missing = set()
    for city in CITIES:
        path = os.path.join(args.input_dir, f"features_{city}_diversity.json")
        with open(path, "r") as f:
            data = json.load(f)
        city_data[city] = data
        missing = []
        for feat in data["features"]:
            geoid = str(feat["properties"]["GEOID"]).zfill(11)
            eth = feat["properties"].get("ethnicity")
            if eth is None:
                missing.append(geoid)
        missing_by_city[city] = missing
        all_missing.update(missing)
        print(f"{city}: {len(missing)} tracts missing ethnicity data")

    print(f"\nTotal unique missing GEOIDs across all cities: {len(all_missing)}")
    if not all_missing:
        print("Nothing to backfill - all tracts already matched.")
        return

    # 2. Get the crosswalk for exactly the missing GEOIDs
    download_relationship_file(args.relationship_file)
    print("\nBuilding crosswalk for missing tracts...")
    crosswalk = build_crosswalk(args.relationship_file, all_missing)
    found_in_crosswalk = sum(1 for g in all_missing if g in crosswalk)
    print(f"Found crosswalk entries for {found_in_crosswalk}/{len(all_missing)} missing tracts")

    # 3. Fetch 2020 data for every 2020 tract referenced by the crosswalk
    all_2020_geoids = set()
    for geoid10, matches in crosswalk.items():
        for geoid20, share in matches:
            all_2020_geoids.add(geoid20)
    print(f"\nFetching 2020 Census data for {len(all_2020_geoids)} crosswalked tracts...")
    data_2020 = fetch_2020_tract_data(list(all_2020_geoids), api_key)
    print(f"Got data for {len(data_2020)} of them")

    # 4. Aggregate (sum) across matched 2020 tracts for each missing 2010 tract
    recovered = {}
    for geoid10, matches in crosswalk.items():
        agg = {"total": 0, POP_VARIABLE: 0}
        for name in ETHNICITY_VARIABLES.values():
            agg[name] = 0
        found_any = False
        for geoid20, share in matches:
            d = data_2020.get(geoid20)
            if d is None:
                continue
            found_any = True
            agg[POP_VARIABLE] += d[POP_VARIABLE]
            for code, name in ETHNICITY_VARIABLES.items():
                agg[name] += d[code]
        if found_any:
            recovered[geoid10] = agg

    print(f"Successfully recovered data for {len(recovered)}/{len(all_missing)} missing tracts")

    # 5. Write back into each city's file, filling only the gaps
    for city in CITIES:
        data = city_data[city]
        filled = 0
        for feat in data["features"]:
            geoid = str(feat["properties"]["GEOID"]).zfill(11)
            if feat["properties"].get("ethnicity") is not None:
                continue  # already had data, don't touch it
            rec = recovered.get(geoid)
            if rec is None:
                continue
            feat["properties"]["Population"] = rec[POP_VARIABLE]
            feat["properties"]["ethnicity"] = {
                name: rec[name] for name in ETHNICITY_VARIABLES.values()
            }
            filled += 1

        in_path = os.path.join(args.input_dir, f"features_{city}_diversity.json")
        out_path = os.path.join(output_dir, f"features_{city}_diversity.json")
        if os.path.abspath(out_path) == os.path.abspath(in_path):
            backup_path = in_path + ".crosswalk.bak"
            if not os.path.exists(backup_path):
                shutil.copy2(in_path, backup_path)

        with open(out_path, "w") as f:
            json.dump(data, f)
        still_missing = len(missing_by_city[city]) - filled
        print(f"{city}: filled {filled} tracts via crosswalk, {still_missing} still unmatched")


if __name__ == "__main__":
    main()

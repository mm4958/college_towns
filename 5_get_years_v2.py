#!/usr/bin/env python3
"""
enrich_with_birth_death.py

Reads CSVs with researcher names, looks up:
- date_of_birth (P569)
- date_of_death (P570)
- ORCID (P496)
- LOC ID (P244)
- affiliation (P108)

Writes enriched CSVs per institution.

Dependencies:
    pip install pandas SPARQLWrapper
"""

import pandas as pd
import time
from SPARQLWrapper import SPARQLWrapper, JSON


# ── CONFIG ────────────────────────────────────────────────
INSTITUTIONS = {
    "mit": {
        "input":  "/home/mm4958/openalex/results/with_nearest_hospital_mit.csv",
        "output": "/home/mm4958/openalex/results/with_death_dates_mit.csv"
    },
    "cornell": {
        "input":  "/home/mm4958/openalex/results/with_nearest_hospital_cornell.csv",
        "output": "/home/mm4958/openalex/results/with_death_dates_cornell.csv"
    },
    "OU": {
        "input":  "/home/mm4958/openalex/results/with_nearest_hospital_ou.csv",
        "output": "/home/mm4958/openalex/results/with_death_dates_ou.csv"
    },
}

NAME_COL = "name"
REQUEST_DELAY_SEC = 1.0

# ── SET UP SPARQL CLIENT (ONCE) ───────────────────────────
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "MITResearchScript/1.0 (mm4958@mit.edu)"

sparql = SPARQLWrapper(SPARQL_ENDPOINT, agent=USER_AGENT)
sparql.setReturnFormat(JSON)


# ── WIKIDATA QUERY ───────────────────────────────────────
def fetch_dates_from_wikidata(label: str):
    query = f"""
        SELECT ?dob ?dod ?orcid ?loc_id ?affiliationLabel WHERE {{
        SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch";
                            wikibase:endpoint "www.wikidata.org";
                            mwapi:search "{label}";
                            mwapi:language "en".
            ?person wikibase:apiOutputItem mwapi:item.
        }}
        ?person wdt:P31 wd:Q5;
                wdt:P569 ?dob.
        OPTIONAL {{ ?person wdt:P570 ?dod. }}
        OPTIONAL {{ ?person wdt:P496 ?orcid. }}
        OPTIONAL {{ ?person wdt:P244 ?loc_id. }}
        OPTIONAL {{ ?person wdt:P108 ?affiliation. }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
    """

    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]

        if not bindings:
            return None, None, None, None, None

        b = bindings[0]
        return (
            b.get("dob", {}).get("value"),
            b.get("dod", {}).get("value"),
            b.get("orcid", {}).get("value"),
            b.get("loc_id", {}).get("value"),
            b.get("affiliationLabel", {}).get("value"),
        )

    except Exception as e:
        print(f"⚠ SPARQL error for '{label}': {e}")
        return None, None, None, None, None


# ── MAIN PER-INSTITUTION ─────────────────────────────────
def main(slug, props):
    print(f"\n=== Processing {slug.upper()} ===")

    df = pd.read_csv(props["input"], dtype=str)
    df.columns = df.columns.str.strip()

    if NAME_COL not in df.columns:
        raise KeyError(
            f"Column '{NAME_COL}' not found in {props['input']}. "
            f"Available columns: {df.columns.tolist()}"
        )

    unique_names = df[NAME_COL].dropna().unique().tolist()
    print(f"→ {len(unique_names)} unique researchers")

    lookup = {}

    for i, name in enumerate(unique_names, start=1):
        print(f"[{i}/{len(unique_names)}] {name}", end="", flush=True)

        dob, dod, orcid, loc_id, affiliation = fetch_dates_from_wikidata(name)
        lookup[name] = {
            "date_of_birth": dob,
            "date_of_death": dod,
            "ORCID": orcid,
            "loc_id": loc_id,
            "affiliation": affiliation,
        }

        print(
            f" → dob={dob!r}, dod={dod!r}, "
            f"ORCID={orcid!r}, loc_id={loc_id!r}"
        )

        time.sleep(REQUEST_DELAY_SEC)

    for col in ["date_of_birth", "date_of_death", "ORCID", "loc_id", "affiliation"]:
        df[col] = df[NAME_COL].map(lambda n: lookup.get(n, {}).get(col))

    df.to_csv(props["output"], index=False)
    print(f"✓ Wrote {len(df)} rows → {props['output']}")


# ── RUN ALL ──────────────────────────────────────────────
if __name__ == "__main__":
    for slug, props in INSTITUTIONS.items():
        main(slug, props)

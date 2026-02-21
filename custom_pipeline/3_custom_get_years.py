"""
Reads a CSV (with researcher names), looks up each researcher’s date of birth (P569)
and date of death (P570) on Wikidata, and writes an enriched CSV with:
    - date_of_birth (ISO)
    - date_of_death (ISO)
    - ORCID, LOC ID, affiliations
"""

import pandas as pd
import asyncio
import aiohttp
import time

# ── CONFIG ───────────────────────────────────────────────
INPUT_CSV = "/home/mm4958/openalex/custom_pipeline/results/with_nearest_hospital_custom.csv"
OUTPUT_CSV = "/home/mm4958/openalex/custom_pipeline/results/with_death_dates_custom.csv"
NAME_COL = "name"
REQUESTS_PER_SEC = 1.0  # Wikidata rate limit
MAX_RETRIES = 5
RETRY_CODES = {429, 500, 502, 503, 504}

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "MITResearchScript/1.0 (mm4958@mit.edu)"}

# ── HELPER FUNCTIONS ─────────────────────────────────────
async def fetch_wikidata(name: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
    """Query Wikidata: search by ORCID if available, else by name."""
    # Construct the SPARQL query
    query = f"""
    SELECT ?dob ?dod ?orcid ?loc_id ?affiliationLabel WHERE {{
        SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch";
                            wikibase:endpoint "www.wikidata.org";
                            mwapi:search "{name}";
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
    url = "https://query.wikidata.org/sparql"
    params = {"query": query, "format": "json"}

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(url, params=params, headers=HEADERS) as resp:
                    if resp.status in RETRY_CODES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    js = await resp.json()
                    bindings = js["results"]["bindings"]
                    if not bindings:
                        return name, None, None, None, None, None
                    b = bindings[0]
                    dob = b["dob"]["value"]
                    dod = b.get("dod", {}).get("value")
                    orcid = b.get("orcid", {}).get("value")
                    loc_id = b.get("loc_id", {}).get("value")
                    affiliation = b.get("affiliationLabel", {}).get("value")
                    return name, dob, dod, orcid, loc_id, affiliation
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"⚠ Failed lookup for '{name}': {e}")
                    return name, None, None, None, None, None
                await asyncio.sleep(2 ** attempt)
    return name, None, None, None, None, None

# ── MAIN ASYNC FUNCTION ──────────────────────────────────
async def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    df.columns = df.columns.str.strip()
    if NAME_COL not in df.columns:
        raise KeyError(f"Column '{NAME_COL}' not found in {INPUT_CSV}")

    unique_names = df[NAME_COL].dropna().unique().tolist() #CHECK FOR DUPLICATE NAMES
    print(f"→ {len(unique_names)} unique researcher names to query.")

    # Semaphore to rate-limit requests
    semaphore = asyncio.Semaphore(REQUESTS_PER_SEC)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_wikidata(name, session, semaphore) for name in unique_names]
        results = []
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            # Respect Wikidata rate limit
            await asyncio.sleep(1.0 / REQUESTS_PER_SEC)

    # Build lookup dict
    lookup = {name: {"date_of_birth": dob, "date_of_death": dod,
                     "ORCID": orcid, "loc_id": loc_id, "affiliation": affiliation}
              for name, dob, dod, orcid, loc_id, affiliation in results}

    # Map results back into DataFrame
    df["date_of_birth"] = df[NAME_COL].map(lambda n: lookup.get(n, {}).get("date_of_birth"))
    df["date_of_death"] = df[NAME_COL].map(lambda n: lookup.get(n, {}).get("date_of_death"))
    df["ORCID"] = df[NAME_COL].map(lambda n: lookup.get(n, {}).get("ORCID"))
    df["loc_id"] = df[NAME_COL].map(lambda n: lookup.get(n, {}).get("loc_id"))
    df["affiliation"] = df[NAME_COL].map(lambda n: lookup.get(n, {}).get("affiliation"))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Done! Wrote {len(df)} rows with birth/death dates to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    asyncio.run(main())

#THis file will get us a timeline of what institutions the author worked at throughout their life. 
#PARALLEL VERSION
import asyncio
import aiohttp
import async_timeout
import pandas as pd
from collections import defaultdict
from aiohttp import ClientSession, ClientResponseError
import math
import time
# ---------------- CONFIG ------------------------
INSTITUTIONS = {
    "mit": {
        "input_profiles_csv": "/home/mm4958/openalex/results/mit_only_affiliations.csv",
        "output_spans_csv":   "mit_author_institution_year_spans.csv"
    },
    "ou": {
        "input_profiles_csv": "/home/mm4958/openalex/results/ou_only_affiliations.csv",
        "output_spans_csv":   "ou_author_institution_year_spans.csv"
    },
    "osu": {
        "input_profiles_csv": "/home/mm4958/openalex/results/osu_only_affiliations.csv",
        "output_spans_csv":   "osu_author_institution_year_spans.csv"
    },
    "dartmouth": {
        "input_profiles_csv": "/home/mm4958/openalex/results/dartmouth_only_affiliations.csv",
        "output_spans_csv":   "dartmouth_author_institution_year_spans.csv"
    },
    "cornell": {
        "input_profiles_csv": "/home/mm4958/openalex/results/cornell_only_affiliations.csv",
        "output_spans_csv":   "cornell_author_institution_year_spans.csv"
    },
    "harvard": {
        "input_profiles_csv": "/home/mm4958/openalex/results/harvard_only_affiliations.csv",
        "output_spans_csv":   "harvard_author_institution_year_spans.csv"
    },
}

PER_PAGE = 200
CONCURRENCY_LIMIT = 3            # how many simultaneous OpenAlex calls
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
REQUEST_DELAY = 1               # polite delay between API pages
API_KEY = "Q7Ou3SucgOyNGJaqU37mFv"
HEADERS = {
    "User-Agent": "MyResearchScraper/1.0 (mailto:mm4958@mit.edu)",
    "Authorization": f"Bearer {API_KEY}"
}
# ---------------- CACHE -------------------------
# Stores results so the same author is never fetched twice
author_cache = {}   # { author_id : { "_profile": ..., (inst_id, inst_name): {years} } }


# ---------------- LOW-LEVEL FETCH w/ RETRIES -------------------------
async def fetch_json(url: str, session: ClientSession, semaphore: asyncio.Semaphore):
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with async_timeout.timeout(45):  # slightly higher
                    async with session.get(url) as r:
                        print(f"DEBUG STATUS {r.status} → {url}")
                        
                        if r.status in RETRY_STATUS_CODES:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        r.raise_for_status()                        
                        return await r.json()

            except Exception as e:
                print(f"!! Unexpected error fetching {url}: {repr(e)}")
                return None

    return None


# ---------------- FETCH ALL WORKS FOR AN AUTHOR -------------------------
async def fetch_author_works(author_id, session, semaphore):
    """Return a dict: { (inst_id, inst_name) : {year: count_of_works} } for this author."""
    if author_id in author_cache and any(k for k in author_cache[author_id] if k != "_profile"):
        return {k: v for k, v in author_cache[author_id].items() if k != "_profile"}

    inst_year_counts = defaultdict(lambda: defaultdict(int))
    cursor = "*"

    while cursor:
        for attempt in range(1, MAX_RETRIES + 1):
            url = (
                "https://api.openalex.org/works"
                f"?filter=authorships.author.id:{author_id}"
                f"&per_page={PER_PAGE}&cursor={cursor}"
            )
            async with semaphore:   # <-- ensures max concurrency
                data = await fetch_json(url, session, semaphore)

            if data is None:
                # Retry if under MAX_RETRIES
                wait = 2 ** attempt
                print(f"Retrying works for {author_id} in {wait}s (attempt {attempt})")
                await asyncio.sleep(wait)
                continue

            # Got data, break out of retry loop
            break

        if not data:
            print(f"DEBUG: No works returned for {author_id} (url={url})")
            break

        if cursor == "*":
            print(f"DEBUG: First works page returned for {author_id}")

        cursor = data["meta"].get("next_cursor")
        results = data.get("results", [])
        if not results:
            break

        for work in results:
            yr = work.get("publication_year")
            if not isinstance(yr, int):
                continue

            for auth in work.get("authorships", []):
                author_full_id = auth.get("author", {}).get("id")
                if isinstance(author_full_id, str) and author_full_id.endswith(author_id):
                    for inst in auth.get("institutions", []):
                        inst_id = inst.get("id")
                        inst_name = inst.get("display_name", "")
                        if inst_id:
                            inst_year_counts[(inst_id, inst_name)][yr] += 1
                    break

        if len(results) < PER_PAGE:
            break

        await asyncio.sleep(REQUEST_DELAY)

    # store in cache
    author_cache.setdefault(author_id, {}).update(inst_year_counts)
    return inst_year_counts


# ---------------- FETCH AUTHOR PROFILE FOR ORCID -------------------------
async def fetch_author_profile(author_id, session, semaphore):
    """Fetch author profile to extract ORCID."""
    if author_id in author_cache and "_profile" in author_cache[author_id]:
        return author_cache[author_id]["_profile"]

    url = f"https://api.openalex.org/authors/{author_id}"
    data = await fetch_json(url, session, semaphore)
    if data:
        print(f"DEBUG: Author profile returned for {author_id} (url={url})")
        author_cache.setdefault(author_id, {})["_profile"] = data

    if not data:
        print(f"DEBUG: No author profile returned for {author_id} (url={url})")
        return None
    
    return data


# ---------------- FETCH BOTH WORKS AND PROFILE -------------------------
async def fetch_author_data(author_id, session, semaphore):
    """Fetch profile first, then works sequentially to avoid 429s."""
    profile = await fetch_author_profile(author_id, session, semaphore)
    works_dict = await fetch_author_works(author_id, session, semaphore)
    
    orcid = None
    domain = field = subfield = ""
    if profile:
        orcid = profile.get("orcid") or profile.get("ids", {}).get("orcid")
        if orcid:
            orcid = orcid.rsplit("/", 1)[-1]
        
        topics = profile.get("topics") or profile.get("ids", {}).get("topics")
        if topics:
            # Find topic with highest count
            top_topic = max(topics, key=lambda t: t.get("count", 0))

            domain = top_topic.get("domain", {}).get("display_name", "")
            field = top_topic.get("field", {}).get("display_name", "")
            subfield = top_topic.get("subfield", {}).get("display_name", "")
        else:
            print("No topics found for this author")

    return works_dict, orcid, domain, field, subfield


# ---------------- SPELL LOGIC -------------------------
def years_to_spells(years, max_gap=5):
    if not years:
        return []
    years = sorted(years)
    spells = []
    start = prev = years[0]
    for y in years[1:]:
        if y - prev <= max_gap:
            prev = y
        else:
            spells.append((start, prev))
            start = prev = y
    spells.append((start, prev))
    return spells


# ---------------- PROCESS ONE INSTITUTION -------------------------
# ---------------- PROCESS ONE INSTITUTION -------------------------
async def process_institution(slug, paths, session, semaphore):
    input_csv  = paths["input_profiles_csv"]
    output_csv = paths["output_spans_csv"]

    try:
        df = pd.read_csv(input_csv, dtype=str)
        # For debugging, sample 1% (remove later)
        sample_size = max(1, math.ceil(len(df) * 0.1))
        df = df.sample(n=sample_size, random_state=42)
    except FileNotFoundError:
        print(f"⚠ Skipping {slug.upper()} — file not found")
        return

    authors = df["author_id"].tolist()
    name_map = dict(zip(df["author_id"], df.get("name", "")))

    print(f"\n--- Processing {slug.upper()} ({len(authors)} authors) ---")

    rows = []

    # LOOP OVER AUTHORS SEQUENTIALLY
    for aurl in authors:
        author_id = aurl.rstrip("/").split("/")[-1]
        print(f"→ Fetching works + profile for {author_id}")
        
        # fetch works + profile sequentially for this author
        inst_dict, orcid, domain, field, subfield = await fetch_author_data(author_id, session, semaphore)

        if not inst_dict:
            print(f"No works found for {aurl}")

        name = name_map.get(aurl, "")
        for (inst_id, inst_name), years in inst_dict.items():
            spells = years_to_spells(years, max_gap=5)
            for i, (start, end) in enumerate(spells, 1):
                years_in_spell = [y for y in range(start, end + 1) if y in years]
                n_active_years = len(years_in_spell)
                n_works = sum(years[y] for y in years_in_spell)
                spell_length = end - start + 1
                rows.append({
                    "author_id": author_id,
                    "name": name,
                    "orcid": orcid,
                    "top_domain": domain,
                    "top_field": field,
                    "top_subfield": subfield,
                    "inst_id": inst_id,
                    "inst_name": inst_name,
                    "spell_num": i,
                    "year_start": start,
                    "year_end": end,
                    "spell_length": spell_length,
                    "n_active_years": n_active_years,
                    "n_works": n_works,
                    "mean_works_per_active_year": n_works / n_active_years if n_active_years else 0,
                    "mean_works_per_calendar_year": n_works / spell_length
                })

        # Optional: polite pause between authors
        await asyncio.sleep(2)   # increase if you still get 429s

    # Build final dataframe
    df_out = pd.DataFrame(rows)
    assert (df_out["n_works"] > 0).all()
    assert df_out["year_start"].le(df_out["year_end"]).all()

    # Summary stats
    spells_per_author = df_out.groupby("author_id")["spell_num"].nunique().describe()
    num_rows = len(df_out)
    summary_df = spells_per_author.reset_index()
    summary_df.columns = ["stat", "value"]
    summary_df = pd.concat([summary_df, pd.DataFrame([{"stat": "total_rows", "value": num_rows}])], ignore_index=True)

    # Save CSVs
    summary_df.to_csv(f"/home/mm4958/openalex/results/{slug}_spell_stats.csv", index=False)         
    df_out.to_csv(f"/home/mm4958/openalex/results/{output_csv}", index=False)
    print(f"[{slug}] Saved {len(df_out)} affiliation spells → {output_csv}")



# ---------------- MAIN DRIVER -------------------------
async def main():
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession(connector=connector, headers=HEADERS) as session:
        for slug, paths in INSTITUTIONS.items():
            await process_institution(slug, paths, session, semaphore)


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")
#This script queries OpenAlex for a list of authors that have been affiliated with a list of
#institutions. It also adds the years they were at the institution.
#Test line
import requests
import asyncio
import aiohttp
import time
import pandas as pd
from aiohttp import ClientSession, ClientResponseError

# CONFIG: map a short slug to its OpenAlex institution ID
INSTITUTIONS = {
    "mit":      "I63966007",   # Massachusetts Institute of Technology
    "ou":       "I8692664",    # University of Oklahoma
    #"osu":      "I115475287",   # Oklahoma State University
    #"dartmouth":"I107672454",  # Dartmouth College
    "cornell":  "I205783295",  # Cornell University
    #"harvard":  "I136199984",  # Harvard University
}
HEADERS = {
    "User-Agent": "MyResearchScraper/1.0 (mailto:mm4958@mit.edu)"
}
YEAR_MIN, YEAR_MAX = 1955, 2025
CHUNK_SIZE = 10  # years per chunk
REQUESTS_PER_SECOND = 5
MAX_REQUESTS = 10
PER_PAGE = 200

async def rate_limited_fetch(sem, session, url, request_times):
    async with sem:
        start = time.time()
        async with session.get(url) as resp:
            resp.raise_for_status()
            result = await resp.json()
        elapsed = time.time() - start
        request_times.append(elapsed)
        # enforce ~REQUESTS_PER_SECOND pacing
        await asyncio.sleep(1 / REQUESTS_PER_SECOND)
        return result

async def fetch_with_cursor(session, sem, inst_id_num, year_start, year_end, request_times):
    cursor = "*"
    all_results = []
    while cursor:

        if len(request_times) >= MAX_REQUESTS:
            print(f"Reached maximum requests limit ({MAX_REQUESTS}). Stopping.")
            break

        url = (
            "https://api.openalex.org/works"
            f"?filter=institutions.id:{inst_id_num},"
            f"publication_year:{year_start}-{year_end}"
            f"&per_page={PER_PAGE}&cursor={cursor}"
        )
        try:
            result = await rate_limited_fetch(sem, session, url, request_times)
            all_results.extend(result.get("results", []))
            cursor = result["meta"].get("next_cursor")

            total_requests = len(request_times)
            total_time = sum(request_times)
            avg_rps = total_requests / total_time if total_time > 0 else 0
            print(f"[{year_start}-{year_end}] Total so far: {len(all_results)}, "
                  f"Requests made: {total_requests}, avg {avg_rps:.2f} req/sec")

        except ClientResponseError as e:
            print(f"[{year_start}-{year_end}] Error {e.status}: {e.message}")
            break

    return all_results

async def process_institution(slug, inst_id_num):
    authors = {}
    inst_url = f"https://openalex.org/{inst_id_num}"
    sem = asyncio.Semaphore(REQUESTS_PER_SECOND)
    request_times = []

    print(f"Processing institution: {inst_id_num}")

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        tasks = []
        for start in range(YEAR_MIN, YEAR_MAX + 1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE - 1, YEAR_MAX)
            tasks.append(
                fetch_with_cursor(session, sem, inst_id_num, start, end, request_times)
            )
        all_chunks = await asyncio.gather(*tasks)

    all_results = [r for chunk in all_chunks for r in chunk]

    # ---- collect years ----
    for work in all_results:
        year = work.get("publication_year")
        if not isinstance(year, int) or not (YEAR_MIN <= year <= YEAR_MAX):
            continue

        for auth in work.get("authorships", []):
            a = auth.get("author", {})
            aid = a.get("id")
            if not aid:
                continue

            for inst in auth.get("institutions", []):
                iid = inst.get("id")
                if iid and iid.lower() == inst_url.lower():
                    
                    rec = authors.setdefault(aid, {
                        "name": a.get("display_name", "Unknown"),
                        "inst_id": iid,
                        "inst_name": inst.get("display_name", "Unknown"),
                        "year_counts": {}
                    })
                    rec["year_counts"][year] = rec["year_counts"].get(year, 0) + 1
    
    rows = []

    for aid, rec in authors.items():
        years = sorted(rec["year_counts"])
        if not years:
            continue

        year_start = min(years)
        year_end = max(years)
        n_active_years = len(years)
        n_works = sum(rec["year_counts"].values())
        calendar_years = year_end - year_start + 1

        rows.append({
            "author_id": aid,
            "name": rec["name"],
            "inst_id": rec["inst_id"],
            "inst_name": rec["inst_name"],
            "year_start": year_start,
            "year_end": year_end,
            "calendar_years": calendar_years,
            "n_active_years": n_active_years,
            "n_works": n_works,
            "mean_works_per_active_year": n_works / n_active_years,
            "mean_works_per_calendar_year": n_works / calendar_years
        })
    # ---- build final dataframe ----
    df = pd.DataFrame(rows)
    assert (df["n_works"] > 0).all()
    assert df["year_start"].le(df["year_end"]).all()
    assert df["n_active_years"].le(df["calendar_years"]).all()
    
    # ---- summary statistics ----
    author_stats = (
        df[["n_works", "n_active_years", "calendar_years"]]
        .describe()
        .reset_index()
    )
    author_stats.rename(columns={"index": "stat"}, inplace=True)
    total_authors = df["author_id"].nunique()
    total_rows = len(df)
    counts_df = pd.DataFrame([
        {"stat": "total_authors", "value": total_authors},
        {"stat": "total_rows", "value": total_rows}
    ])

    summary_df = pd.concat(
        [author_stats, counts_df],
        ignore_index=True
    )
    summary_df.to_csv(
        f"/home/mm4958/openalex/results/{slug}_summary_stats.csv",
        index=False
    )   

    out_fn = f"/home/mm4958/openalex/results/{slug}_only_affiliations.csv"
    df.to_csv(out_fn, index=False)

    print(f"[{slug}] Saved {len(df)} author–institution rows → {out_fn}")
                    
# ── ENTRYPOINT ───────────────────────────────────────────────────────────────
async def main():
    start_time = time.time()
    
    for slug, props in INSTITUTIONS.items():
        await process_institution(slug, props) #once we add more institutions, this needs to be turned into a task so we can run asynchronously, using semaphore to rate limit to 10 req per second
    
    elapsed_time = time.time() - start_time
    print(f"\n Finished in {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    
if __name__ == "__main__":
    asyncio.run(main())
    
#Below is new code so that we can keep track of multiple spells at one instutition








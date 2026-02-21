#This debug script reveals that some later works do not show up in the final list of affiliations and thus that the last year
#of publication can come before the actual final year because sometimes the work does not attribute an instutiton to the author
#so the block of code that stores authorships (included below) skips:
#THIS CODE IS USED TO BUILD PROFILES FOR A CUSTOMIZED LIST OF AUTHORS, JUST INPUT THEIR AUTHOR ID AT THE TOP
'''
for inst in auth.get("institutions", []):
                    inst_id = inst.get("id")
                    inst_name = inst.get("display_name", "")

                    # Add to dict for span calc
                    inst_years[(inst_id, inst_name)].add(pub_year)

                    # Add CSV row
                    rows.append({
                        "work_id": work_id,
                        "publication_year": pub_year,
                        "institution_id": inst_id,
                        "institution_name": inst_name
                    })
'''
import asyncio
import aiohttp
import async_timeout
import pandas as pd
from collections import defaultdict

PER_PAGE = 200
CONCURRENCY_LIMIT = 4
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
REQUEST_DELAY = 0.3
AUTHOR_IDS = [
    "A5113979747", #doyle
    "A5059247979", #gruber
]

# --------------------- LOW-LEVEL FETCH -------------------------

async def fetch_json(url: str, session, semaphore):
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with async_timeout.timeout(30):
                    async with session.get(url) as r:
                        if r.status in RETRY_STATUS_CODES:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        r.raise_for_status()
                        return await r.json()
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"!! Failed fetching {url}: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None

# -------------------- FETCH WORKS FOR ONE AUTHOR --------------------

async def fetch_author_works(author_id, session, semaphore):
    inst_years = defaultdict(set)
    page = 1

    while True:
        url = (
            "https://api.openalex.org/works"
            f"?filter=authorships.author.id:{author_id}"
            f"&per_page={PER_PAGE}&page={page}"
        )

        data = await fetch_json(url, session, semaphore)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            pub_year = work.get("publication_year")
            if not isinstance(pub_year, int):
                continue

            for auth in work.get("authorships", []):
                author_url = auth.get("author", {}).get("id", "")
                if not author_url.endswith(author_id):
                    continue

                for inst in auth.get("institutions", []):
                    inst_id = inst.get("id")
                    inst_name = inst.get("display_name", "")
                    inst_years[(inst_id, inst_name)].add(pub_year)

        if len(results) < PER_PAGE:
            break

        await asyncio.sleep(REQUEST_DELAY)
        page += 1

    return inst_years

# -------------------- FETCH AUTHOR PROFILE --------------------

async def fetch_author_profile(author_id, session, semaphore):
    url = f"https://api.openalex.org/authors/{author_id}"
    data = await fetch_json(url, session, semaphore)

    if not data:
        return None, None

    return (
        data.get("works_count"),
        data.get("cited_by_count"),
        data.get("display_name")
    )

# -------------------------- MAIN ----------------------------

async def main():
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    all_rows = []

    async with aiohttp.ClientSession(connector=connector) as session:
        for author_id in AUTHOR_IDS:

            print(f"\nProcessing author {author_id}...")

            total_works, total_citations, display_name = await fetch_author_profile(
                author_id, session, semaphore
            )

            inst_years = await fetch_author_works(
                author_id, session, semaphore
            )

            for (inst_id, inst_name), years in inst_years.items():
                all_rows.append({
                    "author_id": author_id,
                    "name": display_name,
                    "institution_id": inst_id,
                    "institution_name": inst_name,
                    "year_start": min(years),
                    "year_end": max(years),
                    "total_works": total_works,
                    "total_citations": total_citations
                })

    pd.DataFrame(all_rows).to_csv(
        "/home/mm4958/openalex/custom_pipeline/results/custom_author_institution_year_span.csv", index=False
    )

    print("\n✓ Saved combined affiliation spans → authors_affiliation_spans.csv")

# -------------------- RUN --------------------

if __name__ == "__main__":
    asyncio.run(main())

import pandas as pd
import aiohttp
import asyncio
from pathlib import Path

# ── CONFIG ───────────────────────────────────
INSTITUTIONS = {
    "mit": {
        "input":  "/home/mm4958/openalex/results/with_death_dates_mit.csv",
        "output": "/home/mm4958/openalex/results/mit_final_raw.csv"
    },
    "cornell": {
        "input":  "/home/mm4958/openalex/results/with_death_dates_cornell.csv",
        "output": "/home/mm4958/openalex/results/cornell_final_raw.csv"
    },
    "OU": {
        "input":  "/home/mm4958/openalex/results/with_death_dates_ou.csv",
        "output": "/home/mm4958/openalex/results/ou_final_raw.csv"
    },
}

CONCURRENCY_LIMIT = 5
HEADERS = {"User-Agent": "mm4958@mit.edu"} 

# ── ASYNC FETCH ──────────────────────────────
async def fetch_author_stats(author_id, session, sem):
    url = f"https://api.openalex.org/authors/{author_id}"
    async with sem:
        try:
            async with session.get(url, headers=HEADERS) as r:
                r.raise_for_status()
                js = await r.json()
                return (
                    js.get("works_count"),
                    js.get("cited_by_count"),
                )
        except Exception as e:
            print(f"⚠ Author failed {author_id}: {e}")
            return (None, None)


async def fetch_inst_stats(inst_id, session, sem):
    url = f"https://api.openalex.org/institutions/{inst_id}"
    async with sem:
        try:
            async with session.get(url, headers=HEADERS) as r:
                r.raise_for_status()
                js = await r.json()
                return (js.get("type"),)
        except Exception as e:
            print(f"⚠ Institution failed {inst_id}: {e}")
            return (None,)


async def fetch_all(author_ids, inst_ids):
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession(connector=connector) as session:
        author_tasks = {
            aid: fetch_author_stats(aid, session, sem)
            for aid in author_ids
        }
        inst_tasks = {
            iid: fetch_inst_stats(iid, session, sem)
            for iid in inst_ids
        }

        author_results = await asyncio.gather(*author_tasks.values())
        inst_results   = await asyncio.gather(*inst_tasks.values())

    return (
        dict(zip(author_tasks.keys(), author_results)),
        dict(zip(inst_tasks.keys(), inst_results)),
    )

# ── MAIN PER-INSTITUTION ─────────────────────
def process_institution(slug, props):
    print(f"\nProcessing {slug.upper()}")

    df = pd.read_csv(props["input"], dtype=str)

    df["author_id_short"] = df["author_id"].str.rstrip("/").str.split("/").str[-1]
    df["inst_id_short"]   = df["inst_id"].str.rstrip("/").str.split("/").str[-1]

    unique_authors = df["author_id_short"].unique().tolist()
    unique_insts   = df["inst_id_short"].unique().tolist()

    print(f"• {len(unique_authors)} authors")
    print(f"• {len(unique_insts)} institutions")

    author_stats, inst_stats = asyncio.run(
        fetch_all(unique_authors, unique_insts)
    )

    df["total_works"] = df["author_id_short"].map(
        lambda x: author_stats.get(x, (None, None))[0]
    )
    df["total_citations"] = df["author_id_short"].map(
        lambda x: author_stats.get(x, (None, None))[1]
    )
    df["inst_type"] = df["inst_id_short"].map(
        lambda x: inst_stats.get(x, (None,))[0]
    )

    df.drop(columns=["author_id_short", "inst_id_short"], inplace=True)
    df.to_csv(props["output"], index=False)

    print(f"✓ Saved → {props['output']}")


# ── ENTRYPOINT ───────────────────────────────
if __name__ == "__main__":
    for slug, props in INSTITUTIONS.items():
        process_institution(slug, props)

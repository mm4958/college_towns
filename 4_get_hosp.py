#This script gets the nearest hospital to each institution the author has worked at.
#Note we are only searching for US hospitals, so the code below will ignore non-US institutions for this exercise
#(although) those observatiosn are still kept
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import requests
import io
from sklearn.neighbors import BallTree

# ---------------- CONFIG ----------------
INSTITUTIONS = {
    "mit": {
        "input_csv": "/home/mm4958/openalex/results/mit_author_institution_year_spans.csv",
        "output_csv":   "/home/mm4958/openalex/results/with_nearest_hospital_mit.csv"
    },
    "ou": {
        "input_csv": "/home/mm4958/openalex/results/ou_author_institution_year_spans.csv",
        "output_csv":   "/home/mm4958/openalex/results/with_nearest_hospital_ou.csv"
    },
    #"osu": {
    #    "input_csv": "/home/mm4958/openalex/results/osu_author_institution_year_spans.csv",
    #    "output_csv":   "with_nearest_hospital_osu.csv"
    #},
    #"dartmouth": {
    #    "input_csv": "/home/mm4958/openalex/results/dartmouth_author_institution_year_spans.csv",
    #    "output_csv":   "with_nearest_hospital_dartmouth.csv"
    #},
    "cornell": {
        "input_csv": "/home/mm4958/openalex/results/cornell_author_institution_year_spans.csv",
        "output_csv":   "/home/mm4958/openalex/results/with_nearest_hospital_cornell.csv"
    },
    #"harvard": {
    #    "input_csv": "/home/mm4958/openalex/results/harvard_author_institution_year_spans.csv",
    #    "output_csv":   "with_nearest_hospital_harvard.csv"
    #},
}
HOSPITALS_URL = "https://opendata.arcgis.com/datasets/f36521f6e07f4a859e838f0ad7536898_0.csv"
OA_BASE = "https://api.openalex.org/institutions/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

OA_CONCURRENCY = 5
NOMINATIM_CONCURRENCY = 1
MAX_RETRIES = 5
RETRY_CODES = {429, 500, 502, 503, 504}
HOSP_WAIT = 10


# ---------------- HELPER FUNCTIONS ----------------
async def fetch_json(session, url, semaphore=None):
    """Async fetch with retries and optional semaphore"""
    sem = semaphore or asyncio.Semaphore(1)
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(url) as resp:
                    if resp.status in RETRY_CODES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except Exception:
                if attempt == MAX_RETRIES:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None


async def fetch_inst_coord(inst_id, session, oa_sem):
    key = inst_id.rstrip("/").split("/")[-1]
    url = f"{OA_BASE}{key}"
    print(f"Fetching coordinates for {key} from OpenAlex")
    js = await fetch_json(session, url, oa_sem)
    if not js:
        print(f"[OA] Failed to fetch {key}, got None")
        return inst_id, (None, None)
    
    country = js.get("country_code")
    if country != "US":
        #Skip non-US institutions entirely
        #print(f"[OA] Non-US institution {key}, country={country}")
        return inst_id, (None, None, country)
    geo = js.get("geo", {})
    lat = geo.get("latitude")
    lon = geo.get("longitude")
    if lat is None or lon is None:
        print(f"[OA] No coordinates for {key}, country={country}")
        return inst_id, (None, None, country)

    print(f"[OA] Got coordinates for {key}: lat={lat}, lon={lon}")
    return inst_id, (float(lat), float(lon), country)


async def fallback_nominatim(inst_id, name, session, nom_sem):
    if not name:
        print(f"[NOM] No name for {inst_id}, skipping fallback")
        return inst_id, (None, None)
    
    print(f"[NOM] Fallback for {inst_id} ({name})")
    params = {"q": name, "format": "json", "limit": 1}
    headers = {"User-Agent": "AcademicHealthPanel/1.0"}

    for attempt in range(1, MAX_RETRIES + 1):
        async with nom_sem:
            try:
                async with session.get(NOMINATIM_URL, params=params, headers=headers) as r:
                    if r.status in RETRY_CODES:
                        print(f"[NOM] Retry {attempt} for {inst_id} due to status {r.status}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    r.raise_for_status()
                    js = await r.json()
                    if js:
                        print(f"[NOM] Got fallback coordinates for {inst_id}: lat={js[0]['lat']}, lon={js[0]['lon']}")
                        return inst_id, (float(js[0]["lat"]), float(js[0]["lon"]))
                    return inst_id, (None, None)
            except Exception as e:
                print(f"[NOM] Exception for {inst_id} attempt {attempt}: {e}")
                if attempt == MAX_RETRIES:
                    return inst_id, (None, None)
                await asyncio.sleep(2 ** attempt)
    return inst_id, (None, None)


def download_hospital_list():
    """Download hospital CSV with retry logic"""
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Attempt {attempt}] Downloading hospital list…")
        resp = requests.get(HOSPITALS_URL)
        text = resp.text.strip()

        if text.startswith("{"):
            try:
                msg = resp.json()
            except Exception:
                msg = {"message": "Unknown response"}
            print(f"⚠ Hospital CSV not ready yet: {msg.get('message', msg)}")
            if attempt < MAX_RETRIES:
                import time
                print(f"Waiting {HOSP_WAIT}s before retrying…")
                time.sleep(HOSP_WAIT)
                continue
            else:
                raise RuntimeError("Hospital CSV not ready after maximum retries.")

        hosp = pd.read_csv(io.StringIO(text))
        break

    hosp.columns = hosp.columns.str.strip().str.lower()
    hosp = hosp.rename(columns={"name": "hospital_name", "latitude": "latitude", "longitude": "longitude"})
    hosp = hosp[["hospital_name", "latitude", "longitude"]].dropna()
    hosp["latitude"] = hosp["latitude"].astype(float)
    hosp["longitude"] = hosp["longitude"].astype(float)
    return hosp


async def process_school(input_csv, output_csv, hosp):
    df = pd.read_csv(input_csv, dtype=str)
    unique_insts = df["inst_id"].unique()
    id_to_name = df.set_index("inst_id")["inst_name"].to_dict()

    async with aiohttp.ClientSession() as session:
        # OpenAlex geocoding
        oa_sem = asyncio.Semaphore(OA_CONCURRENCY)
        oa_tasks = [fetch_inst_coord(i, session, oa_sem) for i in unique_insts]
        inst_results = await asyncio.gather(*oa_tasks)
        
        inst_coords = {}
        inst_countries = {}  # track country for each institution

        for inst_id, result in inst_results:
            if len(result) == 3:
                lat, lon, country = result
            else:
                lat = lon = None
                country = None
            inst_coords[inst_id] = (lat, lon)
            inst_countries[inst_id] = country  # save country
            print(f"[DEBUG] {inst_id} -> lat={lat}, lon={lon}, country={country}")
                
        # Fallback Nominatim geocoding (US only)
        nom_sem = asyncio.Semaphore(NOMINATIM_CONCURRENCY)
        need_fallback = [
            (inst_id, id_to_name.get(inst_id, ""))
            for inst_id, (lat, lon) in inst_coords.items()
            if lat is None and inst_countries.get(inst_id) == "US"
        ]
        print(f"[INFO] {len(need_fallback)} US institutions need Nominatim fallback")

        nom_tasks = [fallback_nominatim(inst_id, name, session, nom_sem) for inst_id, name in need_fallback]
        fallback_results = await asyncio.gather(*nom_tasks)
        for inst_id, coord in fallback_results:
            if coord != (None, None):
                inst_coords[inst_id] = coord
                print(f"[NOM] Got fallback coordinates for {inst_id}: {coord}")
       
        # Log US institutions still missing coordinates
        failed_us = [
            k for k, v in inst_coords.items()
            if v == (None, None) and inst_countries.get(k) == "US"
        ]
        if failed_us:
            print(f"⚠ {len(failed_us)} US institutions have missing coordinates: {failed_us}")


    # Nearest hospital lookup
    hosp_rad = np.deg2rad(hosp[["latitude", "longitude"]].values)
    tree = BallTree(hosp_rad, metric="haversine")
    nearest_map = {}
    for inst_id, (lat, lon) in inst_coords.items():
        if lat is None or lon is None:
            nearest_map[inst_id] = {"closest_hospital": "", "hospital_lat": "", "hospital_lon": "", "distance_km": ""}
            continue
        inst_rad = np.deg2rad([[lat, lon]])
        dist_rad, idx = tree.query(inst_rad, k=1)
        rec = hosp.iloc[idx[0][0]]
        nearest_map[inst_id] = {
            "closest_hospital": rec.hospital_name,
            "hospital_lat": rec.latitude,
            "hospital_lon": rec.longitude,
            "distance_km": dist_rad[0][0] * 6371.0
        }

    # Merge + save
    out = df.copy()
    out["closest_hospital"] = out["inst_id"].map(lambda x: nearest_map.get(x, {}).get("closest_hospital", ""))
    out["hospital_lat"] = out["inst_id"].map(lambda x: nearest_map.get(x, {}).get("hospital_lat", ""))
    out["hospital_lon"] = out["inst_id"].map(lambda x: nearest_map.get(x, {}).get("hospital_lon", ""))
    out["distance_km"] = out["inst_id"].map(lambda x: nearest_map.get(x, {}).get("distance_km", ""))
    out.to_csv(output_csv, index=False)
    print(f"✓ Done: wrote {len(out)} rows to {output_csv}")


# ---------------- ENTRYPOINT ----------------
async def main():
    # Download hospital list only once
    hosp = download_hospital_list()
    print(f"✓ Downloaded {len(hosp)} hospitals\n")

    # Process each school sequentially
    for school, cfg in INSTITUTIONS.items():
        print(f"\nProcessing {school.upper()} …")
        await process_school(cfg["input_csv"], cfg["output_csv"], hosp)


if __name__ == "__main__":
    asyncio.run(main())

import pandas as pd
import requests
import io
import time
import numpy as np
from sklearn.neighbors import BallTree

HOSPITALS_URL = "https://opendata.arcgis.com/datasets/f36521f6e07f4a859e838f0ad7536898_0.csv"
MAX_RETRIES = 20        # max attempts
WAIT_SECONDS = 10       # wait between attempts

for attempt in range(1, MAX_RETRIES + 1):
    print(f"[Attempt {attempt}] Downloading hospital list…")
    resp = requests.get(HOSPITALS_URL)
    text = resp.text.strip()
    
    if text.startswith("{"):
        # JSON message, CSV not ready
        try:
            msg = resp.json()
        except Exception:
            msg = {"message": "Unknown response"}
        print(f"⚠ Hospital CSV not ready yet: {msg.get('message', msg)}")
        if attempt < MAX_RETRIES:
            print(f"Waiting {WAIT_SECONDS}s before retrying…")
            time.sleep(WAIT_SECONDS)
            continue
        else:
            raise RuntimeError("Hospital CSV not ready after maximum retries.")
    
    # CSV ready
    hosp = pd.read_csv(io.StringIO(text))
    break

# Check columns and first rows
print("Columns:", hosp.columns.tolist())
print(hosp.head())

# Normalize column names
hosp.columns = hosp.columns.str.strip().str.lower()
hosp = hosp.rename(columns={
    "name": "hospital_name",
    "latitude": "latitude",
    "longitude": "longitude"
})

# Keep only necessary columns
hosp = hosp[["hospital_name", "latitude", "longitude"]].dropna()
hosp["latitude"] = hosp["latitude"].astype(float)
hosp["longitude"] = hosp["longitude"].astype(float)

# Build BallTree
hosp_rad = np.deg2rad(hosp[["latitude", "longitude"]].values)
tree = BallTree(hosp_rad, metric="haversine")
print(f"✓ Loaded {len(hosp)} hospitals and built BallTree.")

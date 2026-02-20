import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_CSV_1 = "/home/mm4958/openalex/results/MIT_author_profiles_extended_f.csv"
INPUT_CSV_2 = "/home/mm4958/openalex/results/with_nearest_hospital_v5.csv"
OUTPUT_CSV  = "/home/mm4958/openalex/results/with_nearest_hospital_v6.csv"

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df1 = pd.read_csv(INPUT_CSV_1, dtype=str)
df2 = pd.read_csv(INPUT_CSV_2, dtype=str)

# ── FILTER TO ONLY ROWS WITH DOB ─────────────────────────────────────────────
df2 = df2[df2["date_of_birth"].notna() & (df2["date_of_birth"].str.strip() != "")]

# ── ORCID MATCH ──────────────────────────────────────────────────────────────
def check_orcid_match(row):
    orcid1 = df1.loc[df1["name"] == row["name"], "orcid"]
    if orcid1.empty or pd.isna(row["ORCID"]) or row["ORCID"].strip() == "":
        return "NA"
    orcid1_val = orcid1.values[0]
    if pd.isna(orcid1_val) or orcid1_val.strip() == "":
        return "NA"
    if orcid1_val.strip() == row["ORCID"].strip():
        return "Y"
    return "N"

df2["orcid_match"] = df2.apply(check_orcid_match, axis=1)

# ── INSTITUTION MATCH ────────────────────────────────────────────────────────
def check_inst_match(row):
    affil = str(row.get("affiliation", "")).strip().lower()
    if affil == "":
        return "NA"
    
    # Get institutions from df1
    matches = df1[df1["name"] == row["name"]]
    if matches.empty:
        return "N"

    current = str(matches.iloc[0].get("current_institutions", "")).lower()
    past = str(matches.iloc[0].get("past_institutions", "")).lower()

    if affil in current:
        return "current"
    elif affil in past:
        return "past"
    else:
        return "N"

df2["inst_match"] = df2.apply(check_inst_match, axis=1)

# ── SAVE OUTPUT ──────────────────────────────────────────────────────────────
df2.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Done! Saved {len(df2)} rows to {OUTPUT_CSV}")








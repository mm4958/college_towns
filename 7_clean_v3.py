#Purpose of this file is to run modify the data to make it less messy and then run an audit to see how it
#compares to an audit of the data when we applied the first attempt at cleaning it. This is done in check.py.
# an audit of the data we get to check for any issues. 
#First load files for all instutitions and append them
import pandas as pd
import gc
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def clean_perfect_overlap(
    df,
    prof_col="author_id",
    lat_col="hospital_lat",
    lon_col="hospital_lon",
    inst_col="inst_name",
    seed=123
):
    rng = np.random.default_rng(seed)
    df = df.copy()
    
    if len(df) == 0:
        print("No perfect overlaps found")
        return df

    indices_to_drop = set()
    affected_authors = set()

    for prof, group in df[df["perfect_overlap"] == 1].groupby(prof_col):
        indices = group.index.tolist()

        # Only relevant if more than one
        while len(indices) > 1:
            i, j = rng.choice(indices, size=2, replace=False)

            row_i, row_j = df.loc[i], df.loc[j]

            lat_i, lon_i = row_i[lat_col], row_i[lon_col]
            lat_j, lon_j = row_j[lat_col], row_j[lon_col]

            inst_i = str(row_i[inst_col]).lower()
            inst_j = str(row_j[inst_col]).lower()

            has_coords_i = not (pd.isna(lat_i) or pd.isna(lon_i))
            has_coords_j = not (pd.isna(lat_j) or pd.isna(lon_j))

            resolved = False

            # ---- Rule 1: coordinates ----
            if not has_coords_i and has_coords_j:
                drop, keep = i, j
                resolved = True
            elif has_coords_i and not has_coords_j:
                drop, keep = j, i
                resolved = True

            # ---- Rule 2: institution name ----
            if not resolved:
                has_univ_i = "university" in inst_i
                has_univ_j = "university" in inst_j

                if has_univ_i and not has_univ_j:
                    drop, keep = j, i
                    resolved = True
                elif has_univ_j and not has_univ_i:
                    drop, keep = i, j
                    resolved = True

            # ---- Rule 3: random tie-break ----
            if not resolved:
                drop = rng.choice([i, j])
                keep = j if drop == i else i
                resolved = True

            # ---- Apply resolution ----
            indices_to_drop.add(drop)
            affected_authors.add(prof)

            indices.remove(drop)

    result = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
    result["perfect_overlap_author"] = result[prof_col].isin(affected_authors).astype(int)

    return result


#Define a function to handle overlapping affiliations
def clean_partial_overlap(
    df,
    prof_col="author_id",
    start_col="year_start",
    end_col="year_end",
    overlap_threshold_years=4,
    seed=123
):
    rng = np.random.default_rng(seed)
    df = df.copy()

    indices_to_drop = set()
    profs_with_drops = set()

    for prof, group in df.groupby(prof_col):
        indices = group.sort_values(start_col).index.tolist()

        changed = True
        while changed:
            changed = False

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    ii, jj = indices[i], indices[j]
                    row_i, row_j = df.loc[ii], df.loc[jj]

                    start_i, end_i = row_i[start_col], row_i[end_col]
                    start_j, end_j = row_j[start_col], row_j[end_col]

                    overlap_start = max(start_i, start_j)
                    overlap_end = min(end_i, end_j)

                    if overlap_start >= overlap_end:
                        continue

                    overlap_years = overlap_end - overlap_start

                    # ---- Rule 1: perfect overlap ----
                    if start_i == start_j and end_i == end_j:
                        continue

                    # ---- Rule 2: subset ----
                    i_subset = start_i >= start_j and end_i <= end_j
                    j_subset = start_j >= start_i and end_j <= end_i

                    if i_subset or j_subset:
                        drop = ii if i_subset else jj
                        indices_to_drop.add(drop)
                        profs_with_drops.add(prof)
                        indices.remove(drop)
                        changed = True
                        break

                    # ---- Rule 3: large partial overlap ----
                    if overlap_years > overlap_threshold_years:
                        dur_i = row_i["appoint_len_w"]
                        dur_j = row_j["appoint_len_w"]

                        if dur_i == dur_j:
                            drop = rng.choice([ii, jj])
                        else:
                            drop = ii if dur_i < dur_j else jj

                        indices_to_drop.add(drop)
                        profs_with_drops.add(prof)
                        indices.remove(drop)
                        changed = True
                        break

                if changed:
                    break

    result = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
    result["partial_overlap_author"] = result[prof_col].isin(profs_with_drops).astype(int)

    return result

def flag_partial_overlap(
    df,
    prof_col="author_id",
    start_col="year_start",
    end_col="year_end"
):
    df = df.copy()
    df["partial_overlap_kept"] = 0

    for prof, group in df.groupby(prof_col):
        rows = group.sort_values(start_col)

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                r1 = rows.iloc[i]
                r2 = rows.iloc[j]

                # any overlap at all
                if min(r1[end_col], r2[end_col]) > max(r1[start_col], r2[start_col]):
                    df.loc[[r1.name, r2.name], "partial_overlap_kept"] = 1

    return df

def mode_string(x):
    x = x.dropna()
    if x.empty:
        return None
    return x.value_counts().idxmax()

#Load data
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 50)
# Load and append files
df1 = pd.read_csv('/home/mm4958/openalex/results/cornell_final_raw.csv')
df2 = pd.read_csv('/home/mm4958/openalex/results/mit_final_raw.csv')
df3 = pd.read_csv('/home/mm4958/openalex/results/ou_final_raw.csv')
comb_df = pd.concat([df1, df2, df3], ignore_index=True)
del df1, df2, df3
gc.collect()
# -------------------------
# Initial stats
# -------------------------
#Drop duplicate affiliations in terms of start and end date and author
unique_affiliation = comb_df.drop_duplicates(
    subset=['inst_id','author_id','year_start','year_end']
).copy()

# Add appointment length
unique_affiliation['appoint_len'] = (
    unique_affiliation['year_end'] - unique_affiliation['year_start'] + 1
)
# Get initial size of dataset and number of unique authors
obs0 = unique_affiliation.shape[0]
authors0 = unique_affiliation['author_id'].nunique()

print(f"\nInitial number of observations: {obs0}")
print(f"Initial number of authors: {authors0}")
# -------------------------
# Raw Summary Stats
# -------------------------
df_copy = unique_affiliation.copy()
df_copy["n_appointments"] = df_copy.groupby("author_id")["author_id"].transform("size")
df_copy['date_of_birth'] = pd.to_datetime(df_copy['date_of_birth'], errors='coerce').dt.year
df_copy['date_of_death'] = pd.to_datetime(df_copy['date_of_death'], errors='coerce').dt.year
# Keep year_start and year_end as-is if they're already integers
df_copy['year_start'] = pd.to_numeric(df_copy['year_start'], errors='coerce')
df_copy['year_end'] = pd.to_numeric(df_copy['year_end'], errors='coerce')
#Create following new vars, lifespan, time between birth and first pub, time between death andlast pub
df_copy['lifespan'] = df_copy['date_of_death'] - df_copy['date_of_birth']
earliest_start = df_copy.groupby('author_id')['year_start'].min().reset_index()
earliest_start = earliest_start.rename(columns={'year_start':'earliest_start'})
latest_end = df_copy.groupby('author_id')['year_end'].max().reset_index()
latest_end = latest_end.rename(columns={'year_end':'latest_end'})
df_copy = pd.merge(df_copy, earliest_start, on='author_id')
df_copy = pd.merge(df_copy, latest_end, on='author_id')
#Create variables measuring time from birth to first pub and time from last pub to death
#Then adjust the latter to discard cases posthomous publications
df_copy['age_first_pub'] = (
    df_copy['earliest_start'] - df_copy['date_of_birth']
)
df_copy['latest_end_adj'] = df_copy[['latest_end', 'date_of_death']].min(axis=1)

df_copy['time_from_last_pub'] = df_copy['date_of_death'] - df_copy['latest_end_adj']

#Collapse data at author level
agg_dict = {
    "distance_km": "mean",  
    "date_of_birth": "first",
    "date_of_death": "first",
    "appoint_len": "mean",
    "lifespan": "first",
    "earliest_start": "first",
    "latest_end_adj": "first",
    "age_first_pub": "first",
    "time_from_last_pub": "first",
    "n_appointments": "first",
    "mean_works_per_active_year": "mean",
    "mean_works_per_calendar_year": "mean",
    "n_active_years":"mean",
    "total_works": "first",
    "n_works": "mean",
    "total_citations": "first",
    "top_domain": mode_string,
    "top_field": mode_string,
    "top_subfield": mode_string,
}

df_raw = (
    df_copy
    .groupby("author_id", as_index=False)
    .agg(agg_dict)
)

#RAW summary stats for all variables at the author level
desc = (df_raw[['distance_km','n_appointments', 'appoint_len','total_citations','total_works', 'n_works','n_active_years',
                'mean_works_per_active_year', 'mean_works_per_calendar_year', 
                'age_first_pub', 'time_from_last_pub', 'lifespan', 'date_of_birth', 'date_of_death',
                'earliest_start', 'latest_end_adj']].describe(percentiles=[0.25, 0.5, 0.75]).round(2))

desc.to_csv("/home/mm4958/openalex/results/df_raw_sum_stats.csv")
df_raw.to_csv("/home/mm4958/openalex/results/df_raw_combined.csv")

topic_vars = ['top_domain', 'top_field', 'top_subfield']
all_top5 = []

for var in topic_vars:
    counts = df_raw[var].value_counts()
    top5 = counts.head(5).rename('count').to_frame()
    top5['percent'] = round(100 * top5['count'] / len(df_raw), 2)
    top5 = top5.reset_index().rename(columns={var: 'category'})  
    top5['variable'] = var  # identify which variable this is
    all_top5.append(top5)
    
# Combine all top5 tables into one
freq_tab_topics = pd.concat(all_top5, ignore_index=True)
# Save to CSV
freq_tab_topics['category'] = freq_tab_topics['category'].str.replace(', ', ' ')
freq_tab_topics.to_csv("/home/mm4958/openalex/results/raw_top5_topics.csv", index=False)
# -------------------------
# Affiliation 
# -------------------------
#remove appointments shorter than 6 years
short_spells = unique_affiliation['appoint_len'] < 6
unique_affiliation = unique_affiliation[~short_spells]
authors1 = unique_affiliation['author_id'].nunique()
obs1 = unique_affiliation.shape[0]
print(f"Dropping appointments <6 years:")
print(f"  Authors dropped: {authors0-authors1} ({(authors0-authors1)/authors0:.2%})")
print(f"  Observations dropped: {obs0-obs1} ({(obs0-obs1)/obs0:.2%})")

#For authors who have more than 1 appointment, drop appointments at instututions that are not classified as being at a healthcare or educational institution
keep_types = {"education", "healthcare"}
multi = unique_affiliation.groupby("author_id")["author_id"].transform("size") > 1
bad   = multi & ~unique_affiliation["inst_type"].isin(keep_types)
unique_affiliation["appt_dropped_by_inst_type"] = bad.groupby(unique_affiliation["author_id"]).transform("any").astype(int)
unique_affiliation = unique_affiliation[~bad].copy()
authors2 = unique_affiliation['author_id'].nunique()
obs2 = unique_affiliation.shape[0]
print(f"For authors that have multiple appointments, dropping those that are not at institutions of type healthcare or education:")
print(f"  Authors affected: {authors1-authors2} ({(authors1-authors2)/authors1:.2%})")
print(f"  Observations affected: {obs1-obs2} ({(obs1-obs2)/obs1:.2%})")

# Winsorize all appointment lengths longer than 50 to 50 years
unique_affiliation["appoint_len_w"] = (
    unique_affiliation["appoint_len"].clip(upper=50)
)
authors_over_50 = unique_affiliation.loc[
    unique_affiliation['appoint_len'] > 50, 'author_id'
]
num_authors_over_50 = authors_over_50.shape[0]
obs_over50 = unique_affiliation[
    unique_affiliation['author_id'].isin(authors_over_50)
].shape[0]
#Create flag for winsorized appointments
unique_affiliation["winsorized_appointment"] = (
    unique_affiliation["appoint_len"] > 50
)
unique_affiliation["winsorized_appointment"] = unique_affiliation["winsorized_appointment"].astype(int)
print(f"Winsorizing appointments >50-year to 50 years:")
print(f"  Authors affected: {num_authors_over_50} ({(num_authors_over_50)/authors2:.2%})")
print(f"  Observations affected: {obs_over50} ({(obs_over50)/obs2:.2%})")

#generate column to keep track of perfect and partial overlaps
unique_affiliation['perfect_overlap'] = 0
unique_affiliation['partial_overlap_kept'] = 0
unique_affiliation['partial_overlap_author'] = 0
# -------------------------
# Clean Partial overlaps 
# -------------------------
df_no_overlap = clean_partial_overlap(unique_affiliation)
obs3 = df_no_overlap.shape[0]
authors3 = df_no_overlap['author_id'].nunique()
print("Dropping appointments that overlap by > 4 years (drop the shorter one)")
print(f"  Authors dropped: {authors2-authors3} ({(authors2-authors3)/authors2:.2%})")
print(f"  Observations dropped: {obs2-obs3} ({(obs2-obs3)/obs2:.2%})")
#Flag remaining partial overlaps
df_no_overlap = flag_partial_overlap(df_no_overlap)
#Compute perfect overlaps
df_no_overlap['perfect_overlap'] = (
    df_no_overlap
    .groupby(['author_id', 'year_start', 'year_end'])['author_id']
    .transform('size') > 1
).astype(int)
perfect_overlap_1 = df_no_overlap['perfect_overlap'].sum()
partial_overlap_kept_n = df_no_overlap['partial_overlap_kept'].sum()
print(f"Number of perfect overlaps: {perfect_overlap_1}")
print(f"Number of partial overlaps that did not meet threshold of 4 years: {partial_overlap_kept_n}")
#download perfect overlaps and partial overlaps to a csv
remaining_overlap = df_no_overlap[
    (df_no_overlap['perfect_overlap'] == 1) | 
    (df_no_overlap['partial_overlap_kept'] == 1)
]
remaining_overlap.to_csv('/home/mm4958/openalex/results/remaining_partial_overlaps_all_perfect_overlaps.csv', index=False)
# -------------------------
# Clean perfect overlaps pairs by 
#(1) Dropping the one with misssing coordinates 
#(2) Removing the one without word "university" in them
#(3) Randomly removing one of the pair
# -------------------------
df_no_overlap["perfect_overlap_author"] = 0
df_affil_clean = clean_perfect_overlap(df_no_overlap)
#Recompute perfect overlaps
df_affil_clean['perfect_overlap'] = (
    df_affil_clean
    .groupby(['author_id', 'year_start', 'year_end'])['author_id']
    .transform('size') > 1
).astype(int)
obs4 = df_affil_clean.shape[0]
authors4 = df_affil_clean['author_id'].nunique()
perfect_overlap_2 = df_affil_clean['perfect_overlap'].sum()
print("Dropping perfect overlaps according to logic above")
print(f"  Authors dropped: {authors3-authors4} ({(authors3-authors4)/authors3:.2%})")
print(f"  Observations dropped: {obs3-obs4} ({(obs3-obs4)/obs3:.2%})")
print(f"Number of perfect overlaps remaining: {perfect_overlap_2} so removed {perfect_overlap_1-perfect_overlap_2}, so ({(perfect_overlap_1-perfect_overlap_2)/perfect_overlap_1:.2%})")
print(f"Number of partial overlaps that did not meet threshold of 4 years: {partial_overlap_kept_n}")
assert perfect_overlap_2 == 0 #Stop code if there are perfect overlaps remaining
# -------------------------
# Now work with death/birth dates
# -------------------------
#Convert variables to date-time objects in years
df_affil_clean['date_of_birth'] = pd.to_datetime(df_affil_clean['date_of_birth'], errors='coerce').dt.year
df_affil_clean['date_of_death'] = pd.to_datetime(df_affil_clean['date_of_death'], errors='coerce').dt.year
# Keep year_start and year_end as-is if they're already integers
df_affil_clean['year_start'] = pd.to_numeric(df_affil_clean['year_start'], errors='coerce')
df_affil_clean['year_end'] = pd.to_numeric(df_affil_clean['year_end'], errors='coerce')
#Create following new vars, lifespan, time between birth and first pub, time between death and last pub
df_affil_clean['lifespan'] = df_affil_clean['date_of_death'] - df_affil_clean['date_of_birth']
earliest_start = df_affil_clean.groupby('author_id')['year_start'].min().reset_index()
earliest_start = earliest_start.rename(columns={'year_start':'earliest_start'})
latest_end = df_affil_clean.groupby('author_id')['year_end'].max().reset_index()
latest_end = latest_end.rename(columns={'year_end':'latest_end'})
df_merged = pd.merge(df_affil_clean, earliest_start, on='author_id')
df_merged = pd.merge(df_merged, latest_end, on='author_id')
#Create variables measuring time from birth to first pub and time from last pub to death
#Then adjust the latter to discard cases posthomous publications
df_merged['age_first_pub'] = (
    df_merged['earliest_start'] - df_merged['date_of_birth']
)
df_merged['latest_end_adj'] = df_merged[['latest_end', 'date_of_death']].min(axis=1)
# Recompute time_from_last_pub using adjusted latest_end
df_merged['time_from_last_pub'] = df_merged['date_of_death'] - df_merged['latest_end_adj']
# -------------------------
# Do some final filtering based on birth and death dates
# -------------------------
#First get some stats for later
lifespan_0 = (df_merged['lifespan']<=15).sum()
pub_b4_birth = (df_merged['age_first_pub']<0).sum()
author_b4_birth = df_merged[df_merged['age_first_pub'] < 0]['author_id'].nunique()
pub_50_birth = (df_merged['age_first_pub']>50).sum()
author_50_birth = df_merged[df_merged['age_first_pub'] > 50]['author_id'].nunique()
#If age_first_pub is <15 or >50 then set to 999, if lifespan is negative then set to 999
bad_age  = (df_merged["age_first_pub"] <= 15) | (df_merged["age_first_pub"] > 50)
bad_life = df_merged["lifespan"] <= 0
# Counts
obs_age  = bad_age.sum()
auth_age = df_merged.loc[bad_age, "author_id"].nunique()
obs_life  = bad_life.sum()
auth_life = df_merged.loc[bad_life, "author_id"].nunique()
obs_affect = obs_age + obs_life
auth_affect = auth_life + auth_age
# Set to missing 
df_merged.loc[bad_age,  "age_first_pub"] = np.nan
df_merged.loc[bad_life, "lifespan"] = np.nan

# Flag columns
df_merged["bad_age_first_pub"] = bad_age.astype(int)
df_merged["bad_lifespan"] = bad_life.astype(int)

print("\nAfter setting to NaN age_first_pub if it was less than 15 years after birth or more than 50 years after birth and setting lifespan to NaN if negative:")
print(f" Observations affected: {obs_affect}  (from {obs4}) ({(obs_affect)/obs4:.2%})")
print(f" Authors affected: {auth_affect}  (from {authors4}) ({(auth_affect)/authors4:.2%})")
print(f" Observations remaining after all changes: {obs4}  (from {obs0}) ({(obs4)/obs0:.2%})")
print(f" Authors remaining after all changes: {authors4}  (from {authors0}) ({(authors4)/authors0:.2%})")
# -------------------------
# Print summary stats
# -------------------------
count_death_date = df_merged[df_merged['date_of_death'].notna() & (df_merged['date_of_death'] != '')]['author_id'].nunique()
count_birth_date = df_merged[df_merged['date_of_birth'].notna() & (df_merged['date_of_birth'] != '')]['author_id'].nunique()
count_birth_death_date = df_merged[
    (df_merged['date_of_death'].notna()) & (df_merged['date_of_death'] != '') & 
    (df_merged['date_of_birth'].notna()) & (df_merged['date_of_birth'] != '')
]['author_id'].nunique()
summary_lifespan = df_merged[['age_first_pub', 'time_from_last_pub', 'lifespan', 'date_of_birth', 'date_of_death','earliest_start', 'latest_end', 'appoint_len_w', "total_works", "total_citations"]].describe(percentiles=[0.25, 0.5, 0.75]).round(2)
print('Summary stats for lifespan variables below:\n', summary_lifespan)
print(f"Non-missing death dates (1 / author): {count_death_date} ({count_death_date/authors3:.2%})")
print(f"Non-missing birth dates (1 / author): {count_birth_date} ({count_birth_date/authors3:.2%})")
print(f"Non-missing birth & death dates (1 / author): {count_birth_death_date} ({count_birth_death_date/authors3:.2%})")
#print(f"Number of observations 15 or less years before earliest start: {pub_b4_birth} ({pub_b4_birth/obs_final:.2%})")
#print(f"Number of authors born 15 or less years before earliest start: {author_b4_birth} ({author_b4_birth/authors_final:.2%})")
#print(f"Number of observations publishing more than 50 years after earliest start: {pub_50_birth} ({pub_50_birth/obs_final:.2%})")
#print(f"Number of authors publishing more than 50 years after earliest start: {author_50_birth} ({author_50_birth/authors_final:.2%})")
#print(f"Dropped this many observations when filtering to only positive lifespans:{lifespan_0}")
# -------------------------
# Additional summary stats
# -------------------------
appointments_per_author = (
    df_merged.groupby('author_id')
    .size()
    .reset_index(name='appointments_per_author')
)

summary_appointments_by_author = appointments_per_author.describe(
    percentiles=[0.25, 0.5, 0.75]
).round(2)

print('\nSummary stats for number of appointments per author below:\n', summary_appointments_by_author)
df_merged.to_csv('/home/mm4958/openalex/results/df_cleaned_final.csv', index=False)
# -------------------------
# Frequency table for domain, field, and subfield
# -------------------------

topic_vars = ['top_domain', 'top_field', 'top_subfield']
all_top5 = []

for var in topic_vars:
    counts = df_merged[var].value_counts()
    top5 = counts.head(5).rename('count').to_frame()
    top5['percent'] = round(100 * top5['count'] / len(df_merged), 2)
    top5 = top5.reset_index().rename(columns={var: 'category'})  
    top5['variable'] = var  # identify which variable this is
    all_top5.append(top5)
    
# Combine all top5 tables into one
freq_tab_topics = pd.concat(all_top5, ignore_index=True)
# Save to CSV
freq_tab_topics['category'] = freq_tab_topics['category'].str.replace(', ', ' ')
freq_tab_topics.to_csv("/home/mm4958/openalex/results/cleaned_top5_topics.csv", index=False)

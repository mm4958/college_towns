import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/home/mm4958/openalex/results/df_cleaned_final.csv")
#Create variable for number of appointments per author
df["n_appointments"] = df.groupby("author_id")["author_id"].transform("size")
#Collapse data at author level
agg_dict = {
    "distance_km": "mean",  
    "date_of_birth": "first",
    "date_of_death": "first",
    "appoint_len_w": "mean",
    "lifespan": "first",
    "earliest_start": "first",
    "latest_end_adj": "first",
    "age_first_pub": "first",
    "time_from_last_pub": "first",
    "n_appointments": "first",
    "n_works": "mean",
    "total_works": "first",
    "total_citations": "first",
    "mean_works_per_active_year": "mean",
    "mean_works_per_calendar_year": "mean",
    "n_active_years":"mean",
    "partial_overlap_author": "max",
    "perfect_overlap_author": "max",
    "bad_age_first_pub": "max",
    "bad_lifespan": "max",
    "winsorized_appointment": "max",
    "appt_dropped_by_inst_type": "max"
}

df_author = (
    df
    .groupby("author_id", as_index=False)
    .agg(agg_dict)
)

#Summary stats for all variables at the author level
desc = (df_author[['distance_km','n_appointments','appoint_len_w','total_citations','total_works','n_works', 
                 'n_active_years','mean_works_per_active_year', 'mean_works_per_calendar_year',
                 'winsorized_appointment', 'partial_overlap_author', 'perfect_overlap_author', 
                 'bad_age_first_pub', 'bad_lifespan', 'appt_dropped_by_inst_type',
                  'age_first_pub', 'time_from_last_pub', 'lifespan', 
                 'date_of_birth', 'date_of_death','earliest_start', 'latest_end_adj']].describe(percentiles=[0.25, 0.5, 0.75]).round(2))

desc.to_csv("/home/mm4958/openalex/results/df_final_sum_stats.csv")
#yvars to plot
y_vars = ["total_works","total_citations","distance_km", "date_of_birth", "date_of_death", 
          "appoint_len_w", "lifespan", "latest_end_adj", "age_first_pub", "time_from_last_pub", "n_appointments",
          "mean_works_per_active_year", "mean_works_per_calendar_year", "n_active_years", "n_works"]
#create cohorts in intervals of 10 years based on earliest appointment date
df_author["start_bin"] = (df_author["earliest_start"] // 10) * 10
df_author["start_bin_label"] = (
    df_author["start_bin"].astype(str)
    + "–"
    + (df_author["start_bin"] + 9).astype(str)
)
#Sanity check start dates
print(df_author["start_bin"].value_counts().sort_index())
#Create label dictionary
y_labels = {
    "distance_km": "Average distance to nearest hospital (km)",
    "date_of_birth": "Birth date",
    "date_of_death": "Death date",
    "appoint_len_w": "Average appointment length (years)",
    "lifespan": "Lifespan (years)",
    "latest_end_adj": "Latest (adjusted) appointment date",
    "age_first_pub": "Age at first publication (years)",
    "time_from_last_pub": "Time from last publcation to death (years)",
    "n_appointments": "Number of appointments in lifetime",
    "total_works": "Total works",
    "total_citations": "Total citations",
    "mean_works_per_active_year": "Mean works per active year",
    "mean_works_per_calendar_year": "Mean works per calendar year",
    'n_active_years': "Mean number of active years",
    "n_works":"Mean number of works across observed spells"}

order = sorted(df_author["start_bin"].unique())
fig_dir = "/home/mm4958/openalex/graphs"# Count number of authors per cohort
cohort_counts = (
    df_author
    .groupby("start_bin")
    .size()
    .reindex(order)
)

# ---- WINSORIZE at 1% ----
#for y in y_vars:
#    lower = df_author[y].quantile(0.01)
#    upper = df_author[y].quantile(0.99)
#    df_author[y] = df_author[y].clip(lower=lower, upper=upper)
# ------------------------

for y in y_vars:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df_author,
        x="start_bin",
        y=y,
        order=order
    )
    
    label = y_labels.get(y, y)

    plt.xlabel("Earliest appointment date (10-year cohorts)")
    plt.ylabel(label)
    plt.title(f"{label} by earliest appointment date")
    
    # ---- ADD N LABELS ----
    bin_q3 = (
        df_author
        .groupby("start_bin")[y]
        .quantile(0.75)  # 75th percentile
        .reindex(order)
    )
    
    variable_counts = (
        df_author[df_author[y].notna()]
        .groupby("start_bin")
        .size()
        .reindex(order, fill_value=0)
    )

    for i, bin_val in enumerate(order):
        n = variable_counts.loc[bin_val]
        y_pos = bin_q3.loc[bin_val]
        ax.text(
            i,
            y_pos,
            f"n = {n}",
            ha="center",
            va="top",
            fontsize=9,
            color="orange"
        )
    # ----------------------
    # ---- FIX Y-AXIS LIMITS ----
    y_min = df_author[y].min()
    y_max = df_author[y].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.02 * y_range, y_max + 0.1 * y_range) 
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/boxplot_{y}_by_start_bin.png", dpi=300)
    plt.close()
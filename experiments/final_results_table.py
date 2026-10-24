import pandas as pd
import os

INPUT_FILE = "results/multi_workload_results.csv"
OUTPUT_FILE = "results/final_results_table.csv"

METRICS = [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "cpu_utilization",
    "throughput"
]

LOWER_IS_BETTER = {
    "waiting_time": True,
    "turnaround_time": True,
    "response_time": True,
    "context_switches": True,
    "cpu_utilization": False,
    "throughput": False
}

RL_SCHEDULERS = {
    "Q_Learning_RR",
    "DQN_RR",
    "DDQN_RR",
    "D3QN_RR",
    "PPO_RR"
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("FINAL RESULTS TABLE")
print("=" * 70)

print(f"Total observations: {len(df)}")
print(f"Schedulers: {df['scheduler'].nunique()}")
print(f"Workloads: {df['workload_type'].nunique()}")
print(f"Seeds: {df['seed'].nunique()}")


# ============================================================
# CONVERT METRICS TO NUMERIC
# ============================================================

for metric in METRICS:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")


# Check for missing values
print("\nMissing values:")
print(df[METRICS].isna().sum())


# ============================================================
# CALCULATE MEAN PERFORMANCE
# ============================================================

summary = (
    df.groupby("scheduler", as_index=False)[METRICS]
    .mean()
)


# ============================================================
# CALCULATE RANKS
# ============================================================

print("\nCalculating metric ranks...")

rank_columns = []

for metric in METRICS:

    rank_column = metric + "_rank"

    summary[rank_column] = summary[metric].rank(
        ascending=LOWER_IS_BETTER[metric],
        method="average"
    )

    rank_columns.append(rank_column)


# ============================================================
# CALCULATE OVERALL RANK
# ============================================================

summary["overall_rank_score"] = summary[rank_columns].mean(axis=1)

summary["overall_rank"] = summary["overall_rank_score"].rank(
    ascending=True,
    method="average"
)


# ============================================================
# CATEGORY
# ============================================================

summary["category"] = summary["scheduler"].apply(
    lambda x: "RL-Based"
    if x in RL_SCHEDULERS
    else "Traditional"
)


# ============================================================
# SORT
# ============================================================

summary = summary.sort_values(
    by=["overall_rank_score", "scheduler"]
).reset_index(drop=True)


# ============================================================
# ROUND NUMBERS
# ============================================================

numeric_columns = summary.select_dtypes(
    include="number"
).columns

summary[numeric_columns] = summary[numeric_columns].round(4)


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs("results", exist_ok=True)

summary.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL PERFORMANCE
# ============================================================

print("\n")
print("FINAL SCHEDULER PERFORMANCE")
print("=" * 70)

display_columns = [
    "scheduler",
    "category",
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "cpu_utilization",
    "throughput",
    "overall_rank_score",
    "overall_rank"
]

print(
    summary[display_columns].to_string(index=False)
)


# ============================================================
# FINAL RANKING
# ============================================================

print("\n")
print("FINAL OVERALL RANKING")
print("=" * 70)

ranking = summary[
    [
        "scheduler",
        "category",
        "overall_rank_score",
        "overall_rank"
    ]
]

print(
    ranking.to_string(index=False)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION")
print("=" * 70)

if summary["overall_rank_score"].isna().any():
    print("ERROR: NaN values still exist in overall ranking.")
else:
    print("PASS: No NaN values in overall ranking.")

if summary["overall_rank"].isna().any():
    print("ERROR: NaN values still exist in rank.")
else:
    print("PASS: No NaN values in rank.")


print("\n")
print(f"Saved: {OUTPUT_FILE}")

print("\nFINAL RESULTS TABLE CREATED SUCCESSFULLY!")
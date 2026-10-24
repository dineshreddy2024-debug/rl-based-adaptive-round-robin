import pandas as pd
import os

# ============================================================
# FILES
# ============================================================

INPUT_FILE = "results/multi_workload_results.csv"

OUTPUT_DIR = "results"
PLOT_DIR = "results/plots"


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# SCHEDULER GROUPS
# ============================================================

RL_SCHEDULERS = [
    "Q_Learning_RR",
    "DQN_RR",
    "DDQN_RR",
    "D3QN_RR",
    "PPO_RR"
]

TRADITIONAL_SCHEDULERS = [
    "FCFS",
    "SJF",
    "Fixed_RR",
    "Dynamic_RR"
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

for metric in METRICS:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")


print("=" * 75)
print("FINAL PAPER RESULTS")
print("=" * 75)

print(f"Total observations : {len(df)}")
print(f"Schedulers         : {df['scheduler'].nunique()}")
print(f"Workloads          : {df['workload_type'].nunique()}")
print(f"Seeds              : {df['seed'].nunique()}")


# ============================================================
# 1. OVERALL PERFORMANCE TABLE
# ============================================================

overall = (
    df.groupby("scheduler")[METRICS]
    .mean()
    .reset_index()
)

overall["category"] = overall["scheduler"].apply(
    lambda x: "RL-Based"
    if x in RL_SCHEDULERS
    else "Traditional"
)

overall_file = os.path.join(
    OUTPUT_DIR,
    "paper_overall_performance.csv"
)

overall.to_csv(overall_file, index=False)


# ============================================================
# 2. OVERALL RANKING
# ============================================================

ranking = overall.copy()

rank_columns = []

for metric in METRICS:

    rank_col = metric + "_rank"

    ranking[rank_col] = ranking[metric].rank(
        ascending=LOWER_IS_BETTER[metric],
        method="average"
    )

    rank_columns.append(rank_col)


ranking["overall_rank_score"] = ranking[
    rank_columns
].mean(axis=1)

ranking["overall_rank"] = ranking[
    "overall_rank_score"
].rank(
    ascending=True,
    method="average"
)

ranking = ranking.sort_values(
    ["overall_rank_score", "scheduler"]
)

ranking_file = os.path.join(
    OUTPUT_DIR,
    "paper_overall_ranking.csv"
)

ranking.to_csv(ranking_file, index=False)


# ============================================================
# 3. RL WORKLOAD WINNERS
# ============================================================

rl_df = df[
    df["scheduler"].isin(RL_SCHEDULERS)
].copy()

workload_summary = (
    rl_df.groupby(
        ["workload_type", "scheduler"]
    )[METRICS]
    .mean()
    .reset_index()
)


# Rank RL schedulers separately for every workload
for metric in METRICS:

    workload_summary[
        metric + "_rank"
    ] = workload_summary.groupby(
        "workload_type"
    )[metric].rank(
        ascending=LOWER_IS_BETTER[metric],
        method="average"
    )


workload_rank_columns = [
    metric + "_rank"
    for metric in METRICS
]

workload_summary["overall_rank_score"] = (
    workload_summary[
        workload_rank_columns
    ].mean(axis=1)
)


winners = workload_summary.loc[
    workload_summary.groupby(
        "workload_type"
    )["overall_rank_score"].idxmin()
].copy()

winners = winners.sort_values("workload_type")


winner_columns = [
    "workload_type",
    "scheduler",
    "overall_rank_score"
]

winners = winners[winner_columns]

winner_file = os.path.join(
    OUTPUT_DIR,
    "final_rl_workload_winners.csv"
)

winners.to_csv(
    winner_file,
    index=False
)


# ============================================================
# 4. RL WIN COUNTS
# ============================================================

win_counts = (
    winners["scheduler"]
    .value_counts()
    .rename_axis("scheduler")
    .reset_index(name="workload_wins")
)

win_counts_file = os.path.join(
    OUTPUT_DIR,
    "final_rl_win_counts.csv"
)

win_counts.to_csv(
    win_counts_file,
    index=False
)


# ============================================================
# 5. PAPER-READY SUMMARY TABLE
# ============================================================

paper_table = ranking[
    [
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
].copy()

paper_table = paper_table.round(4)

paper_table_file = os.path.join(
    OUTPUT_DIR,
    "paper_ready_results_table.csv"
)

paper_table.to_csv(
    paper_table_file,
    index=False
)


# ============================================================
# 6. CREATE PAPER FIGURES
# ============================================================

import matplotlib.pyplot as plt


for metric in [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "throughput"
]:

    plot_data = overall.sort_values(
        metric,
        ascending=LOWER_IS_BETTER[metric]
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        plot_data["scheduler"],
        plot_data[metric]
    )

    plt.xlabel("Scheduler")
    plt.ylabel(metric.replace("_", " ").title())

    plt.title(
        "Overall " +
        metric.replace("_", " ").title()
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    filename = os.path.join(
        PLOT_DIR,
        "paper_" + metric + ".png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. RL WORKLOAD WIN GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    win_counts["scheduler"],
    win_counts["workload_wins"]
)

plt.xlabel("RL Scheduler")
plt.ylabel("Number of Workloads Won")

plt.title(
    "RL Scheduler Performance Across Workloads"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

win_plot = os.path.join(
    PLOT_DIR,
    "paper_rl_workload_wins.png"
)

plt.savefig(
    win_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("FINAL OVERALL RANKING")
print("=" * 75)

print(
    ranking[
        [
            "scheduler",
            "category",
            "overall_rank_score",
            "overall_rank"
        ]
    ].round(4).to_string(index=False)
)


print("\n")
print("BEST RL SCHEDULER BY WORKLOAD")
print("=" * 75)

print(
    winners.round(4).to_string(index=False)
)


print("\n")
print("RL WORKLOAD WIN COUNTS")
print("=" * 75)

print(
    win_counts.to_string(index=False)
)


# ============================================================
# FILE SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FILES CREATED")
print("=" * 75)

print(overall_file)
print(ranking_file)
print(winner_file)
print(win_counts_file)
print(paper_table_file)

print("\nPlots:")

for metric in [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "throughput"
]:
    print(
        os.path.join(
            PLOT_DIR,
            "paper_" + metric + ".png"
        )
    )

print(
    os.path.join(
        PLOT_DIR,
        "paper_rl_workload_wins.png"
    )
)

print("\n")
print("=" * 75)
print("FINAL PAPER RESULTS CREATED SUCCESSFULLY!")
print("=" * 75)
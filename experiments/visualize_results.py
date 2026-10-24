import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "multi_workload_results.csv"
PLOT_DIR = RESULTS_DIR / "plots"

PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("Loaded results:")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
metrics = [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "cpu_utilization",
    "throughput"
]


# ---------------------------------------------------------
# Workload types
# ---------------------------------------------------------
workloads = sorted(df["workload_type"].unique())

print("\nWorkloads:")
for workload in workloads:
    print("-", workload)


# ---------------------------------------------------------
# Schedulers
# ---------------------------------------------------------
schedulers = sorted(df["scheduler"].unique())

print("\nSchedulers:")
for scheduler in schedulers:
    print("-", scheduler)


# =========================================================
# 1. OVERALL PERFORMANCE
# =========================================================

summary = (
    df.groupby("scheduler")[metrics]
    .mean()
)

print("\nOverall Scheduler Performance:")
print(summary)


for metric in metrics:

    values = summary[metric].sort_values()

    plt.figure(figsize=(12, 6))

    values.plot(kind="bar")

    plt.title(
        f"Overall {metric.replace('_', ' ').title()}"
    )

    plt.xlabel("Scheduler")

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    output_file = PLOT_DIR / f"{metric}.png"

    plt.savefig(output_file, dpi=300)

    plt.close()

    print(f"Saved: {output_file}")


# =========================================================
# 2. WORKLOAD-WISE PERFORMANCE
# =========================================================

print("\nCreating workload-wise graphs...")


for metric in metrics:

    workload_summary = (
        df.groupby(["workload_type", "scheduler"])[metric]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(14, 7))

    workload_summary.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title(
        f"{metric.replace('_', ' ').title()} Across Workloads"
    )

    plt.xlabel("Workload Type")

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.xticks(rotation=45, ha="right")

    plt.legend(
        title="Scheduler",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        f"workload_{metric}.png"
    )

    plt.savefig(output_file, dpi=300)

    plt.close()

    print(f"Saved: {output_file}")


# =========================================================
# 3. RL-ONLY COMPARISON
# =========================================================

rl_schedulers = [
    "Q_Learning_RR",
    "DQN_RR",
    "DDQN_RR",
    "D3QN_RR",
    "PPO_RR"
]

rl_df = df[
    df["scheduler"].isin(rl_schedulers)
]


print("\nCreating RL-only comparisons...")


for metric in metrics:

    rl_summary = (
        rl_df
        .groupby(["workload_type", "scheduler"])[metric]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(14, 7))

    rl_summary.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title(
        f"RL Scheduler Comparison - "
        f"{metric.replace('_', ' ').title()}"
    )

    plt.xlabel("Workload Type")

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.xticks(rotation=45, ha="right")

    plt.legend(
        title="RL Scheduler",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        f"rl_workload_{metric}.png"
    )

    plt.savefig(output_file, dpi=300)

    plt.close()

    print(f"Saved: {output_file}")


# =========================================================
# 4. NORMALIZED OVERALL SCORE
# =========================================================

normalized = summary.copy()


lower_better = [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches"
]

higher_better = [
    "cpu_utilization",
    "throughput"
]


for metric in lower_better:

    min_value = normalized[metric].min()
    max_value = normalized[metric].max()

    if max_value != min_value:

        normalized[metric] = (
            (max_value - normalized[metric])
            / (max_value - min_value)
        )

    else:

        normalized[metric] = 1.0


for metric in higher_better:

    min_value = normalized[metric].min()
    max_value = normalized[metric].max()

    if max_value != min_value:

        normalized[metric] = (
            (normalized[metric] - min_value)
            / (max_value - min_value)
        )

    else:

        normalized[metric] = 1.0


normalized["overall_score"] = (
    normalized[metrics].mean(axis=1)
)


normalized = normalized.sort_values(
    "overall_score",
    ascending=False
)


print("\nNormalized Overall Score:")

print(
    normalized["overall_score"]
)


plt.figure(figsize=(12, 6))

normalized["overall_score"].plot(
    kind="bar"
)

plt.title(
    "Overall Normalized Scheduler Performance"
)

plt.xlabel("Scheduler")

plt.ylabel(
    "Normalized Score"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

output_file = (
    PLOT_DIR /
    "overall_scheduler_score.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.close()

print(
    f"Saved: {output_file}"
)


# =========================================================
# FINISHED
# =========================================================

print("\n========================================")
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("========================================")

print(f"\nGraphs are located in:")
print(PLOT_DIR)

# =========================================================
# 5. BEST RL SCHEDULER FOR EACH WORKLOAD
# =========================================================

print("\n========================================")
print("BEST RL SCHEDULER FOR EACH WORKLOAD")
print("========================================")


rl_workload_summary = (
    rl_df
    .groupby(["workload_type", "scheduler"])[metrics]
    .mean()
)


# Calculate rank for each metric
ranked = rl_workload_summary.copy()

for metric in lower_better:
    ranked[metric + "_rank"] = (
        ranked.groupby(level=0)[metric]
        .rank(ascending=True)
    )


for metric in higher_better:
    ranked[metric + "_rank"] = (
        ranked.groupby(level=0)[metric]
        .rank(ascending=False)
    )


rank_columns = [
    metric + "_rank"
    for metric in metrics
]


ranked["overall_rank"] = (
    ranked[rank_columns].mean(axis=1)
)


# Find best RL scheduler for every workload
best_rl = (
    ranked
    .reset_index()
    .sort_values(
        ["workload_type", "overall_rank"]
    )
    .groupby("workload_type")
    .first()
)


print("\nBest RL scheduler for each workload:\n")

for workload, row in best_rl.iterrows():

    print(
        f"{workload:20s} -> "
        f"{row['scheduler']:15s} "
        f"(rank = {row['overall_rank']:.2f})"
    )


# Save results
best_rl_output = (
    RESULTS_DIR /
    "best_rl_scheduler_per_workload.csv"
)


best_rl.reset_index().to_csv(
    best_rl_output,
    index=False
)


print(
    f"\nSaved: {best_rl_output}"
)


# =========================================================
# 6. HOW OFTEN EACH RL SCHEDULER WINS
# =========================================================

win_counts = (
    best_rl["scheduler"]
    .value_counts()
)


print("\n========================================")
print("RL SCHEDULER WIN COUNTS")
print("========================================")

print(win_counts)


win_output = (
    RESULTS_DIR /
    "rl_scheduler_win_counts.csv"
)


win_counts.to_csv(
    win_output,
    header=["workload_wins"]
)


print(
    f"\nSaved: {win_output}"
)

# =========================================================
# 7. RL WORKLOAD WIN COUNT GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

win_counts.sort_values(ascending=False).plot(
    kind="bar"
)

plt.title(
    "Number of Workloads Won by Each RL Scheduler"
)

plt.xlabel("RL Scheduler")

plt.ylabel("Number of Workloads")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

output_file = (
    PLOT_DIR /
    "rl_workload_win_count.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.close()

print(
    f"\nSaved: {output_file}"
)
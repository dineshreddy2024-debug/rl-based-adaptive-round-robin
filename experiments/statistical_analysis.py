import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


# ============================================================
# LOAD DATA
# ============================================================

INPUT_FILE = "results/multi_workload_results.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

print(f"Total observations: {len(df)}")
print(f"Schedulers: {df['scheduler'].nunique()}")
print(f"Workloads: {df['workload_type'].nunique()}")
print(f"Seeds: {df['seed'].nunique()}")


SCHEDULERS = sorted(df["scheduler"].unique())

METRICS = [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "cpu_utilization",
    "throughput"
]


# ============================================================
# OVERALL MEAN AND STD
# ============================================================

summary = (
    df.groupby("scheduler")[METRICS]
      .agg(["mean", "std"])
)

summary.columns = [
    "_".join(col)
    for col in summary.columns
]

summary = summary.reset_index()


# ============================================================
# RANK SCHEDULERS
# ============================================================

# Lower is better for these metrics
LOWER_IS_BETTER = [
    "waiting_time_mean",
    "turnaround_time_mean",
    "response_time_mean",
    "context_switches_mean"
]

# Higher is better
HIGHER_IS_BETTER = [
    "cpu_utilization_mean",
    "throughput_mean"
]


rank_columns = []

for column in LOWER_IS_BETTER:

    rank_column = column.replace(
        "_mean",
        "_rank"
    )

    summary[rank_column] = (
        summary[column]
        .rank(
            ascending=True,
            method="min"
        )
    )

    rank_columns.append(rank_column)


for column in HIGHER_IS_BETTER:

    rank_column = column.replace(
        "_mean",
        "_rank"
    )

    summary[rank_column] = (
        summary[column]
        .rank(
            ascending=False,
            method="min"
        )
    )

    rank_columns.append(rank_column)


# ============================================================
# OVERALL RANK
# ============================================================

summary["overall_rank_score"] = (
    summary[rank_columns]
    .mean(axis=1)
)

summary["overall_rank"] = (
    summary["overall_rank_score"]
    .rank(
        ascending=True,
        method="min"
    )
)


# ============================================================
# PRINT RANKING
# ============================================================

print("\n" + "=" * 80)
print("OVERALL SCHEDULER RANKING")
print("=" * 80)

ranking = summary[
    [
        "scheduler",
        "overall_rank_score",
        "overall_rank"
    ]
].sort_values("overall_rank")

print(
    ranking.to_string(index=False)
)


# ============================================================
# FRIEDMAN TEST
# ============================================================

print("\n" + "=" * 80)
print("FRIEDMAN TEST")
print("=" * 80)

friedman_results = []


# Each workload + seed is one paired block
df["block"] = (
    df["workload_type"].astype(str)
    + "_"
    + df["seed"].astype(str)
)


for metric in METRICS:

    pivot = df.pivot(
        index="block",
        columns="scheduler",
        values=metric
    )

    # Keep only complete blocks
    pivot = pivot.dropna()

    arrays = [
        pivot[scheduler].values
        for scheduler in SCHEDULERS
    ]

    statistic, p_value = friedmanchisquare(
        *arrays
    )

    friedman_results.append({

        "metric": metric,

        "friedman_statistic":
            statistic,

        "p_value":
            p_value,

        "significant":
            p_value < 0.05
    })

    print(
        f"{metric:<20} "
        f"statistic={statistic:.4f} "
        f"p={p_value:.6f}"
    )


friedman_df = pd.DataFrame(
    friedman_results
)


# ============================================================
# PAIRWISE WILCOXON TESTS
# ============================================================

print("\n" + "=" * 80)
print("PAIRWISE WILCOXON TESTS")
print("=" * 80)


pairwise_results = []

for metric in METRICS:

    pivot = df.pivot(
        index="block",
        columns="scheduler",
        values=metric
    ).dropna()

    for i in range(len(SCHEDULERS)):

        for j in range(i + 1, len(SCHEDULERS)):

            scheduler_a = SCHEDULERS[i]
            scheduler_b = SCHEDULERS[j]

            a = pivot[scheduler_a].values
            b = pivot[scheduler_b].values

            try:

                statistic, p_value = wilcoxon(
                    a,
                    b,
                    zero_method="wilcox",
                    alternative="two-sided"
                )

            except ValueError:

                statistic = np.nan
                p_value = np.nan

            pairwise_results.append({

                "metric": metric,

                "scheduler_a":
                    scheduler_a,

                "scheduler_b":
                    scheduler_b,

                "wilcoxon_statistic":
                    statistic,

                "p_value":
                    p_value,

                "significant":
                    p_value < 0.05
                    if not np.isnan(p_value)
                    else False
            })


pairwise_df = pd.DataFrame(
    pairwise_results
)


# ============================================================
# SAVE RESULTS
# ============================================================

summary_file = (
    "results/statistical_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


friedman_file = (
    "results/friedman_test.csv"
)

friedman_df.to_csv(
    friedman_file,
    index=False
)


pairwise_file = (
    "results/pairwise_wilcoxon.csv"
)

pairwise_df.to_csv(
    pairwise_file,
    index=False
)


# ============================================================
# BEST SCHEDULER PER METRIC
# ============================================================

print("\n" + "=" * 80)
print("BEST SCHEDULER PER METRIC")
print("=" * 80)


best_results = []


for metric in METRICS:

    if metric in [
        "waiting_time",
        "turnaround_time",
        "response_time",
        "context_switches"
    ]:

        best_row = summary.loc[
            summary[f"{metric}_mean"].idxmin()
        ]

    else:

        best_row = summary.loc[
            summary[f"{metric}_mean"].idxmax()
        ]

    best_results.append({

        "metric": metric,

        "best_scheduler":
            best_row["scheduler"],

        "mean":
            best_row[f"{metric}_mean"],

        "std":
            best_row[f"{metric}_std"]
    })


best_df = pd.DataFrame(
    best_results
)


print(
    best_df.to_string(index=False)
)


best_file = (
    "results/best_scheduler_per_metric.csv"
)

best_df.to_csv(
    best_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 80)

print("\nCreated:")

print(summary_file)
print(friedman_file)
print(pairwise_file)
print(best_file)
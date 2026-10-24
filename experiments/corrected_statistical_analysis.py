import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "results/multi_workload_results.csv"

METRICS = [
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches",
    "cpu_utilization",
    "throughput"
]


# Lower is better
LOWER_IS_BETTER = {
    "waiting_time",
    "turnaround_time",
    "response_time",
    "context_switches"
}


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)

    adjusted = np.empty(len(p_values))

    for rank, index in enumerate(order):

        adjusted[index] = (
            p_values[index] *
            (len(p_values) - rank)
        )

    # Ensure adjusted p-values are monotonic
    for i in range(1, len(order)):

        previous = order[i - 1]
        current = order[i]

        adjusted[current] = max(
            adjusted[current],
            adjusted[previous]
        )

    adjusted = np.minimum(adjusted, 1.0)

    return adjusted


# ============================================================
# EFFECT SIZE
# ============================================================

def wilcoxon_effect_size(a, b):

    differences = np.asarray(a) - np.asarray(b)

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    abs_diff = np.abs(differences)

    ranks = pd.Series(abs_diff).rank(
        method="average"
    ).to_numpy()

    w_plus = np.sum(
        ranks[differences > 0]
    )

    w_minus = np.sum(
        ranks[differences < 0]
    )

    denominator = w_plus + w_minus

    if denominator == 0:
        return 0.0

    return (
        (w_plus - w_minus)
        / denominator
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["block"] = (
    df["workload_type"].astype(str)
    + "_"
    + df["seed"].astype(str)
)

schedulers = sorted(
    df["scheduler"].unique()
)

print("=" * 80)
print("HOLM-CORRECTED PAIRWISE STATISTICAL ANALYSIS")
print("=" * 80)

print(f"Schedulers : {len(schedulers)}")
print(f"Workloads  : {df['workload_type'].nunique()}")
print(f"Seeds      : {df['seed'].nunique()}")


# ============================================================
# PAIRWISE TESTS
# ============================================================

results = []


for metric in METRICS:

    print("\n" + "-" * 80)
    print(f"METRIC: {metric}")
    print("-" * 80)

    pivot = df.pivot(
        index="block",
        columns="scheduler",
        values=metric
    ).dropna()

    raw_results = []

    for i in range(len(schedulers)):

        for j in range(i + 1, len(schedulers)):

            scheduler_a = schedulers[i]
            scheduler_b = schedulers[j]

            a = pivot[scheduler_a].values
            b = pivot[scheduler_b].values

            try:

                statistic, p_value = wilcoxon(
                    a,
                    b,
                    zero_method="wilcox",
                    alternative="two-sided"
                )

                effect = wilcoxon_effect_size(
                    a,
                    b
                )

            except ValueError:

                statistic = np.nan
                p_value = np.nan
                effect = np.nan

            raw_results.append({

                "metric": metric,

                "scheduler_a":
                    scheduler_a,

                "scheduler_b":
                    scheduler_b,

                "wilcoxon_statistic":
                    statistic,

                "raw_p_value":
                    p_value,

                "effect_size":
                    effect
            })


    # --------------------------------------------------------
    # Holm correction
    # --------------------------------------------------------

    valid_indices = [
        i for i, x in enumerate(raw_results)
        if not np.isnan(x["raw_p_value"])
    ]

    valid_p_values = [
        raw_results[i]["raw_p_value"]
        for i in valid_indices
    ]

    corrected = holm_correction(
        valid_p_values
    )

    for index, corrected_p in zip(
        valid_indices,
        corrected
    ):

        raw_results[index][
            "holm_adjusted_p_value"
        ] = corrected_p

        raw_results[index][
            "significant_0.05"
        ] = corrected_p < 0.05


    results.extend(raw_results)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = (
    "results/corrected_pairwise_wilcoxon.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT SIGNIFICANT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("SIGNIFICANT PAIRWISE COMPARISONS")
print("=" * 80)

significant = results_df[
    results_df["significant_0.05"] == True
].copy()

if len(significant) == 0:

    print(
        "No statistically significant "
        "pairwise differences after Holm correction."
    )

else:

    print(
        significant[
            [
                "metric",
                "scheduler_a",
                "scheduler_b",
                "raw_p_value",
                "holm_adjusted_p_value",
                "effect_size"
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL METRIC RANKING
# ============================================================

summary = (
    df.groupby("scheduler")[METRICS]
    .mean()
)


ranking_rows = []


for metric in METRICS:

    if metric in LOWER_IS_BETTER:

        ranked = summary[metric].sort_values(
            ascending=True
        )

    else:

        ranked = summary[metric].sort_values(
            ascending=False
        )

    for rank, (scheduler, value) in enumerate(
        ranked.items(),
        start=1
    ):

        ranking_rows.append({

            "metric": metric,

            "scheduler": scheduler,

            "mean": value,

            "rank": rank
        })


ranking_df = pd.DataFrame(
    ranking_rows
)

ranking_file = (
    "results/final_metric_ranking.csv"
)

ranking_df.to_csv(
    ranking_file,
    index=False
)


# ============================================================
# AVERAGE RANK ACROSS METRICS
# ============================================================

average_rank = (
    ranking_df
    .groupby("scheduler")["rank"]
    .mean()
    .sort_values()
    .reset_index()
)

average_rank.columns = [
    "scheduler",
    "average_rank"
]

average_rank["overall_rank"] = (
    average_rank["average_rank"]
    .rank(
        method="min",
        ascending=True
    )
)

overall_file = (
    "results/final_overall_ranking.csv"
)

average_rank.to_csv(
    overall_file,
    index=False
)


# ============================================================
# PRINT FINAL RANKING
# ============================================================

print("\n" + "=" * 80)
print("FINAL OVERALL RANKING")
print("=" * 80)

print(
    average_rank.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("CORRECTED STATISTICAL ANALYSIS COMPLETE")
print("=" * 80)

print("\nCreated:")

print(output_file)
print(ranking_file)
print(overall_file)
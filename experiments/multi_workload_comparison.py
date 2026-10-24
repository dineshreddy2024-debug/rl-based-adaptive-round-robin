import os
import csv
import contextlib
import io
import pandas as pd

from experiments.baseline_comparison import run_baseline_comparison


# ============================================================
# CONFIGURATION
# ============================================================

WORKLOAD_TYPES = [
    "short_cpu",
    "long_cpu",
    "mixed",
    "interactive",
    "batch",
    "bursty",
    "high_arrival",
    "low_arrival",
    "priority_mixed",
    "heavy_tail",
    "uniform",
    "stress"
]

SEEDS = [42, 43, 44, 45, 46]

NUM_PROCESSES = 20
FIXED_QUANTUM = 4


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_multi_workload_experiment():

    os.makedirs("results", exist_ok=True)

    all_results = []

    total_experiments = len(WORKLOAD_TYPES) * len(SEEDS)
    experiment_number = 0

    print("=" * 80)
    print("MULTI-WORKLOAD SCHEDULER EVALUATION")
    print("=" * 80)

    print(f"Workload types : {len(WORKLOAD_TYPES)}")
    print(f"Random seeds   : {len(SEEDS)}")
    print(f"Schedulers     : 9")
    print(f"Processes      : {NUM_PROCESSES}")
    print(f"Total tests    : {total_experiments}")
    print("=" * 80)

    # --------------------------------------------------------
    # Loop through workload types
    # --------------------------------------------------------

    for workload_type in WORKLOAD_TYPES:

        for seed in SEEDS:

            experiment_number += 1

            print(
                f"\n[{experiment_number}/{total_experiments}] "
                f"Workload={workload_type}, Seed={seed}"
            )

            # ------------------------------------------------
            # Suppress the detailed output from
            # baseline_comparison.py
            # ------------------------------------------------

            with contextlib.redirect_stdout(io.StringIO()):

                results = run_baseline_comparison(
                    num_processes=NUM_PROCESSES,
                    workload_type=workload_type,
                    seed=seed,
                    fixed_quantum=FIXED_QUANTUM
                )

            # ------------------------------------------------
            # Store every scheduler result
            # ------------------------------------------------

            for scheduler, metrics in results["schedulers"].items():

                all_results.append({

                    "workload_type": workload_type,

                    "seed": seed,

                    "num_processes": NUM_PROCESSES,

                    "scheduler": scheduler,

                    "waiting_time":
                        metrics["waiting_time"],

                    "turnaround_time":
                        metrics["turnaround_time"],

                    "response_time":
                        metrics["response_time"],

                    "context_switches":
                        metrics["context_switches"],

                    "cpu_utilization":
                        metrics["cpu_utilization"],

                    "throughput":
                        metrics["throughput"]
                })

            print(
                f"Completed: {workload_type} "
                f"(seed {seed})"
            )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    raw_file = "results/multi_workload_results.csv"

    df = pd.DataFrame(all_results)

    df.to_csv(
        raw_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("RAW RESULTS SAVED")
    print("=" * 80)

    print(raw_file)

    # ========================================================
    # CALCULATE MEAN AND STANDARD DEVIATION
    # ========================================================

    metric_columns = [
        "waiting_time",
        "turnaround_time",
        "response_time",
        "context_switches",
        "cpu_utilization",
        "throughput"
    ]

    grouped = (
        df
        .groupby(
            ["workload_type", "scheduler"]
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten column names

    grouped.columns = [

        "_".join(col).strip("_")
        if isinstance(col, tuple)
        else col

        for col in grouped.columns
    ]

    summary_file = (
        "results/multi_workload_summary.csv"
    )

    grouped.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("WORKLOAD SUMMARY SAVED")
    print("=" * 80)

    print(summary_file)

    # ========================================================
    # OVERALL RESULTS ACROSS ALL WORKLOADS
    # ========================================================

    overall = (
        df
        .groupby("scheduler")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    overall.columns = [

        "_".join(col).strip("_")
        if isinstance(col, tuple)
        else col

        for col in overall.columns
    ]

    overall_file = (
        "results/overall_scheduler_summary.csv"
    )

    overall.to_csv(
        overall_file,
        index=False
    )

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY SAVED")
    print("=" * 80)

    print(overall_file)

    # ========================================================
    # PRINT OVERALL COMPARISON
    # ========================================================

    print("\n" + "=" * 100)
    print("OVERALL SCHEDULER PERFORMANCE")
    print("=" * 100)

    print(
        overall[
            [
                "scheduler",
                "waiting_time_mean",
                "turnaround_time_mean",
                "response_time_mean",
                "context_switches_mean",
                "cpu_utilization_mean",
                "throughput_mean"
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # CHECK CPU UTILIZATION
    # ========================================================

    max_cpu = df["cpu_utilization"].max()

    print("\n" + "=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    print(
        f"Maximum CPU utilization observed: "
        f"{max_cpu:.2f}%"
    )

    if max_cpu > 100.0:

        print(
            "WARNING: CPU utilization exceeded 100%."
        )

    else:

        print(
            "PASS: CPU utilization is within valid range."
        )

    print("\n" + "=" * 80)
    print("MULTI-WORKLOAD EXPERIMENT COMPLETE")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_multi_workload_experiment()
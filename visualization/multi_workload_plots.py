import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

file = RESULTS / "overall_scheduler_summary.csv"

df = pd.read_csv(file)

print("\nLoaded overall results:")
print(df)


# ============================================================
# Plot mean ± standard deviation
# ============================================================

def plot_metric(
    mean_column,
    std_column,
    ylabel,
    filename
):

    plt.figure(figsize=(12, 6))

    plt.bar(
        df["scheduler"],
        df[mean_column],
        yerr=df[std_column],
        capsize=5
    )

    plt.xlabel("Scheduler")
    plt.ylabel(ylabel)

    plt.title(
        f"{ylabel}: Mean ± Standard Deviation"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output = RESULTS / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output}")


# ============================================================
# Generate research graphs
# ============================================================

plot_metric(
    "waiting_time_mean",
    "waiting_time_std",
    "Average Waiting Time",
    "multi_workload_waiting_time.png"
)

plot_metric(
    "turnaround_time_mean",
    "turnaround_time_std",
    "Average Turnaround Time",
    "multi_workload_turnaround_time.png"
)

plot_metric(
    "response_time_mean",
    "response_time_std",
    "Average Response Time",
    "multi_workload_response_time.png"
)

plot_metric(
    "context_switches_mean",
    "context_switches_std",
    "Context Switches",
    "multi_workload_context_switches.png"
)

plot_metric(
    "cpu_utilization_mean",
    "cpu_utilization_std",
    "CPU Utilization (%)",
    "multi_workload_cpu_utilization.png"
)

plot_metric(
    "throughput_mean",
    "throughput_std",
    "Throughput",
    "multi_workload_throughput.png"
)


print("\nAll multi-workload graphs created successfully.")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Project root
ROOT = Path(__file__).resolve().parent.parent

csv_file = ROOT / "results" / "scheduler_comparison.csv"
results_dir = ROOT / "results"

# Read CSV
df = pd.read_csv(csv_file)

print("\nLoaded scheduler comparison:")
print(df)


# ============================================================
# Function to create a comparison graph
# ============================================================

def create_plot(column, ylabel, filename, lower_is_better=True):

    plt.figure(figsize=(12, 6))

    plt.bar(
        df["scheduler"],
        df[column]
    )

    plt.xlabel("Scheduler")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    output = results_dir / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {output}")


# ============================================================
# Create all graphs
# ============================================================

create_plot(
    "waiting_time",
    "Average Waiting Time",
    "waiting_time_comparison.png"
)

create_plot(
    "turnaround_time",
    "Average Turnaround Time",
    "turnaround_time_comparison.png"
)

create_plot(
    "response_time",
    "Average Response Time",
    "response_time_comparison.png"
)

create_plot(
    "context_switches",
    "Context Switches",
    "context_switches_comparison.png"
)

create_plot(
    "cpu_utilization",
    "CPU Utilization (%)",
    "cpu_utilization_comparison.png"
)

create_plot(
    "throughput",
    "Throughput",
    "throughput_comparison.png"
)


print("\nAll comparison graphs created successfully.")
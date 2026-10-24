import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "corrected_pairwise_wilcoxon.csv"
OUTPUT_FILE = RESULTS_DIR / "rl_vs_traditional_significance.csv"


# ---------------------------------------------------------
# Load statistical results
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print("Loaded statistical results:")
print(f"Rows: {len(df)}")


# ---------------------------------------------------------
# RL and traditional schedulers
# ---------------------------------------------------------
rl_schedulers = {
    "Q_Learning_RR",
    "DQN_RR",
    "DDQN_RR",
    "D3QN_RR",
    "PPO_RR"
}

traditional_schedulers = {
    "FCFS",
    "SJF",
    "Fixed_RR",
    "Dynamic_RR"
}


# ---------------------------------------------------------
# Find scheduler column names
# ---------------------------------------------------------
print("\nColumns:")
print(df.columns.tolist())


# The corrected statistical file contains the two
# schedulers being compared.
scheduler_columns = [
    column
    for column in df.columns
    if column.lower() in [
        "scheduler_a",
        "scheduler_b",
        "scheduler1",
        "scheduler2"
    ]
]

print("\nScheduler columns detected:")
print(scheduler_columns)


# ---------------------------------------------------------
# Determine actual column names
# ---------------------------------------------------------
if "scheduler_a" in df.columns:
    scheduler_a = "scheduler_a"
    scheduler_b = "scheduler_b"
elif "scheduler1" in df.columns:
    scheduler_a = "scheduler1"
    scheduler_b = "scheduler2"
else:
    raise ValueError(
        "Could not find scheduler comparison columns."
    )


# ---------------------------------------------------------
# Find adjusted p-value column
# ---------------------------------------------------------
p_columns = [
    column
    for column in df.columns
    if "adjusted" in column.lower()
    and "p" in column.lower()
]

if not p_columns:
    raise ValueError(
        "Could not find adjusted p-value column."
    )

p_column = p_columns[0]

print(f"\nUsing adjusted p-value column: {p_column}")


# ---------------------------------------------------------
# Extract RL vs traditional comparisons
# ---------------------------------------------------------
rows = []

for _, row in df.iterrows():

    s1 = row[scheduler_a]
    s2 = row[scheduler_b]

    is_rl_traditional = (
        (s1 in rl_schedulers and s2 in traditional_schedulers)
        or
        (s2 in rl_schedulers and s1 in traditional_schedulers)
    )

    if not is_rl_traditional:
        continue

    adjusted_p = row[p_column]

    if pd.isna(adjusted_p):
        significance = "Not testable"
    elif adjusted_p < 0.05:
        significance = "Significant"
    else:
        significance = "Not significant"

    rows.append({
        "metric": row["metric"],
        "scheduler_1": s1,
        "scheduler_2": s2,
        "adjusted_p_value": adjusted_p,
        "significance": significance
    })


# ---------------------------------------------------------
# Create result dataframe
# ---------------------------------------------------------
result = pd.DataFrame(rows)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("RL VS TRADITIONAL SIGNIFICANCE")
print("========================================")

print(result.to_string(index=False))

print(
    f"\nSaved: {OUTPUT_FILE}"
)


# ---------------------------------------------------------
# Count significant comparisons
# ---------------------------------------------------------
significant_count = (
    result["significance"] == "Significant"
).sum()

total_count = len(result)

print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    f"Significant comparisons: "
    f"{significant_count}/{total_count}"
)

print(
    f"Non-significant comparisons: "
    f"{total_count - significant_count}/{total_count}"
)
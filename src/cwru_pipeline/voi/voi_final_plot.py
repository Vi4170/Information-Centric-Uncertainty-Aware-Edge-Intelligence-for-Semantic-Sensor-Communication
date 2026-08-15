import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# VALUE OF INFORMATION - FINAL RESULTS PLOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "voi"
    / "voi_optimal_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "voi"
    / "voi_final_plot.png"
)


print("=" * 70)
print("VALUE OF INFORMATION - FINAL RESULTS PLOT")
print("=" * 70)


# ------------------------------------------------------------
# Check input
# ------------------------------------------------------------

print("\nChecking required files...")

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{METRICS_FILE}"
    )

print("VoI metrics found.")


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

print("\nLoading VoI results...")

with open(METRICS_FILE, "r") as f:
    data = json.load(f)


edge_accuracy = data["edge_accuracy"]
cloud_accuracy = data["cloud_accuracy"]

recommended = data["recommended_operating_point"]

recommended_threshold = recommended["threshold"]
recommended_accuracy = recommended["final_accuracy"]
recommended_saving = recommended["communication_savings"]

threshold_results = data["all_threshold_results"]


# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------

thresholds = [
    item["threshold"]
    for item in threshold_results
]

accuracies = [
    item["final_accuracy"] * 100
    for item in threshold_results
]

communication_savings = [
    item["communication_savings"] * 100
    for item in threshold_results
]


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\nFinal results:")
print(f"Edge accuracy:              {edge_accuracy * 100:.2f}%")
print(f"Cloud accuracy:             {cloud_accuracy * 100:.2f}%")
print(
    f"Recommended threshold:      {recommended_threshold:.2f}"
)
print(
    f"Recommended accuracy:       {recommended_accuracy * 100:.2f}%"
)
print(
    f"Communication saving:       {recommended_saving * 100:.2f}%"
)


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

print("\nCreating final plot...")


fig, ax1 = plt.subplots(figsize=(10, 6))


# Accuracy curve

ax1.plot(
    thresholds,
    accuracies,
    marker="o",
    linewidth=2,
    label="VoI Accuracy"
)

ax1.axhline(
    edge_accuracy * 100,
    linestyle="--",
    linewidth=1.5,
    label=f"Edge Accuracy ({edge_accuracy * 100:.2f}%)"
)

ax1.axhline(
    cloud_accuracy * 100,
    linestyle=":",
    linewidth=1.5,
    label=f"Cloud Accuracy ({cloud_accuracy * 100:.2f}%)"
)


# Recommended operating point

ax1.scatter(
    [recommended_threshold],
    [recommended_accuracy * 100],
    s=120,
    marker="*",
    zorder=5,
    label=(
        f"Recommended Point "
        f"({recommended_accuracy * 100:.2f}%, "
        f"{recommended_saving * 100:.1f}% saving)"
    )
)


ax1.set_xlabel("Uncertainty Threshold")
ax1.set_ylabel("Diagnostic Accuracy (%)")
ax1.set_title(
    "Value of Information: Accuracy vs Uncertainty Threshold"
)

ax1.grid(True, alpha=0.3)


# Communication saving on second axis

ax2 = ax1.twinx()

ax2.plot(
    thresholds,
    communication_savings,
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Communication Saving"
)

ax2.set_ylabel("Communication Saving (%)")


# Combined legend

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)


fig.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\nFinal plot saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("FINAL VOI PLOT COMPLETE")
print("=" * 70)
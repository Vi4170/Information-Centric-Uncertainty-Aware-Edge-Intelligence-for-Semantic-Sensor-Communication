import json
from pathlib import Path

import numpy as np


# ============================================================
# VALUE OF INFORMATION - OPTIMAL OPERATING POINT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "voi"

DECISION_FILE = RESULTS_DIR / "voi_decision_results.npz"
METRICS_FILE = RESULTS_DIR / "voi_decision_metrics.json"

OUTPUT_FILE = RESULTS_DIR / "voi_optimal_results.json"


print("=" * 70)
print("VALUE OF INFORMATION - OPTIMAL OPERATING POINT")
print("=" * 70)


# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

print("\nChecking required files...")

if not DECISION_FILE.exists():
    raise FileNotFoundError(
        f"Missing file:\n{DECISION_FILE}"
    )

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Missing file:\n{METRICS_FILE}"
    )

print("Decision results found.")
print("Decision metrics found.")


# ------------------------------------------------------------
# Load metrics
# ------------------------------------------------------------

print("\nLoading VoI decision metrics...")

with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)


threshold_results = metrics["threshold_results"]


# ------------------------------------------------------------
# Define useful targets
# ------------------------------------------------------------

EDGE_ACCURACY = metrics["edge_accuracy"]
CLOUD_ACCURACY = metrics["cloud_accuracy"]


print("\nBase performance:")
print(f"Edge accuracy:  {EDGE_ACCURACY:.4f}")
print(f"Cloud accuracy: {CLOUD_ACCURACY:.4f}")


# ------------------------------------------------------------
# Find best points
# ------------------------------------------------------------

results = []

for threshold_string, data in threshold_results.items():

    results.append({
        "threshold": data["threshold"],
        "transmission_rate": data["transmission_rate"],
        "communication_savings": data["communication_savings"],
        "final_accuracy": data["final_accuracy"],
        "weighted_f1": data["weighted_f1"],
        "accuracy_gain": data["accuracy_gain"],
    })


# ------------------------------------------------------------
# Best accuracy
# ------------------------------------------------------------

best_accuracy = max(
    results,
    key=lambda x: x["final_accuracy"]
)


# ------------------------------------------------------------
# Best F1
# ------------------------------------------------------------

best_f1 = max(
    results,
    key=lambda x: x["weighted_f1"]
)


# ------------------------------------------------------------
# Best accuracy while saving communication
# ------------------------------------------------------------

# We want the highest accuracy among operating points
# that actually save communication.

saving_points = [
    x for x in results
    if x["communication_savings"] > 0
]

best_saving_accuracy = max(
    saving_points,
    key=lambda x: x["final_accuracy"]
)


# ------------------------------------------------------------
# Best point with >= 90% communication savings
# ------------------------------------------------------------

points_90 = [
    x for x in results
    if x["communication_savings"] >= 0.90
]

if points_90:
    best_90 = max(
        points_90,
        key=lambda x: x["final_accuracy"]
    )
else:
    best_90 = None


# ------------------------------------------------------------
# Best point with >= 80% communication savings
# ------------------------------------------------------------

points_80 = [
    x for x in results
    if x["communication_savings"] >= 0.80
]

if points_80:
    best_80 = max(
        points_80,
        key=lambda x: x["final_accuracy"]
    )
else:
    best_80 = None


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST OPERATING POINTS")
print("=" * 70)

print("\n1. BEST OVERALL ACCURACY")
print(f"   Threshold:           {best_accuracy['threshold']:.2f}")
print(f"   Accuracy:            {best_accuracy['final_accuracy']:.4f}")
print(f"   Accuracy:            {best_accuracy['final_accuracy'] * 100:.2f}%")
print(f"   Transmission rate:   {best_accuracy['transmission_rate'] * 100:.1f}%")
print(f"   Communication save:  {best_accuracy['communication_savings'] * 100:.1f}%")


print("\n2. BEST WEIGHTED F1")
print(f"   Threshold:           {best_f1['threshold']:.2f}")
print(f"   Weighted F1:         {best_f1['weighted_f1']:.4f}")
print(f"   Accuracy:            {best_f1['final_accuracy'] * 100:.2f}%")
print(f"   Transmission rate:   {best_f1['transmission_rate'] * 100:.1f}%")
print(f"   Communication save:  {best_f1['communication_savings'] * 100:.1f}%")


print("\n3. BEST ACCURACY WITH SOME COMMUNICATION SAVING")
print(f"   Threshold:           {best_saving_accuracy['threshold']:.2f}")
print(f"   Accuracy:            {best_saving_accuracy['final_accuracy'] * 100:.2f}%")
print(f"   Transmission rate:   {best_saving_accuracy['transmission_rate'] * 100:.1f}%")
print(f"   Communication save:  {best_saving_accuracy['communication_savings'] * 100:.1f}%")


if best_80:
    print("\n4. BEST POINT WITH >= 80% COMMUNICATION SAVINGS")
    print(f"   Threshold:           {best_80['threshold']:.2f}")
    print(f"   Accuracy:            {best_80['final_accuracy'] * 100:.2f}%")
    print(f"   Transmission rate:   {best_80['transmission_rate'] * 100:.1f}%")
    print(f"   Communication save:  {best_80['communication_savings'] * 100:.1f}%")


if best_90:
    print("\n5. BEST POINT WITH >= 90% COMMUNICATION SAVINGS")
    print(f"   Threshold:           {best_90['threshold']:.2f}")
    print(f"   Accuracy:            {best_90['final_accuracy'] * 100:.2f}%")
    print(f"   Transmission rate:   {best_90['transmission_rate'] * 100:.1f}%")
    print(f"   Communication save:  {best_90['communication_savings'] * 100:.1f}%")


# ------------------------------------------------------------
# Simple recommendation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RECOMMENDED OPERATING POINT")
print("=" * 70)

# For the project, choose the highest-accuracy point
# that still saves at least 50% communication.

points_50 = [
    x for x in results
    if x["communication_savings"] >= 0.50
]

if points_50:

    recommended = max(
        points_50,
        key=lambda x: x["final_accuracy"]
    )

    print("\nRecommended threshold:")
    print(f"  {recommended['threshold']:.2f}")

    print("\nExpected performance:")
    print(
        f"  Accuracy:             "
        f"{recommended['final_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Transmission rate:    "
        f"{recommended['transmission_rate'] * 100:.1f}%"
    )

    print(
        f"  Communication saving: "
        f"{recommended['communication_savings'] * 100:.1f}%"
    )

    print(
        f"  Accuracy gain vs edge:"
        f" {recommended['accuracy_gain'] * 100:+.2f}%"
    )

else:

    recommended = None

    print("\nNo operating point provides 50% or more communication savings.")


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

output = {
    "edge_accuracy": EDGE_ACCURACY,
    "cloud_accuracy": CLOUD_ACCURACY,

    "best_overall_accuracy": best_accuracy,
    "best_weighted_f1": best_f1,
    "best_accuracy_with_communication_saving": best_saving_accuracy,

    "best_80_percent_saving": best_80,
    "best_90_percent_saving": best_90,

    "recommended_operating_point": recommended,

    "all_threshold_results": results
}


with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)


print("\nResults saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("OPTIMAL VOI ANALYSIS COMPLETE")
print("=" * 70)
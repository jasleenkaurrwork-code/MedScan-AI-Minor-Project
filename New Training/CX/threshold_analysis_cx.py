import os
"""
Threshold Analysis for CX (Chest X-Ray)
- Reads MC Dropout results CSV
- Tests multiple thresholds
- Finds optimal threshold for 95%+ selective accuracy
- Reports all metrics
"""

import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(SCRIPT_DIR, "MC_results_CX_binary.csv"

def analyze(csv_path):
    df = pd.read_csv(csv_path)

    print("=" * 65)
    print("CX BINARY - THRESHOLD ANALYSIS")
    print("=" * 65)
    print("Total samples:", len(df))
    print("Uncertainty range: [{:.6f}, {:.6f}]".format(
        df["Uncertainty"].min(), df["Uncertainty"].max()
    ))
    print("Uncertainty mean:   {:.6f}".format(df["Uncertainty"].mean()))
    print("Uncertainty median: {:.6f}".format(df["Uncertainty"].median()))
    print()

    baseline_acc = (df["Label"] == df["Prediction"]).mean()
    print("Baseline MC Dropout Accuracy: {:.2f}%".format(baseline_acc * 100))
    print()

    # --- Class-wise accuracy ---
    print("Class-wise accuracy:")
    for cls in sorted(df["Label"].unique()):
        cls_data = df[df["Label"] == cls]
        cls_acc = (cls_data["Label"] == cls_data["Prediction"]).mean()
        print("  Class {}: {:.2f}% ({}/{})".format(
            cls, cls_acc * 100,
            (cls_data["Label"] == cls_data["Prediction"]).sum(),
            len(cls_data)
        ))
    print()

    # --- Threshold sweep ---
    unc_min = df["Uncertainty"].min()
    unc_max = df["Uncertainty"].max()

    thresholds = sorted(set(
        list(np.linspace(unc_min, unc_max, 30)) +
        list(np.percentile(df["Uncertainty"], [10, 20, 30, 40, 50, 60, 70, 80, 90]))
    ))

    header = "{:<14} {:<10} {:<10} {:<12} {:<12} {:<12}".format(
        "Threshold", "Accepted", "Rejected", "Coverage", "Sel. Acc", "Improvement"
    )
    print(header)
    print("-" * 68)

    best_thresh_95 = None
    best_coverage_95 = 0

    for t in thresholds:
        selected = df[df["Uncertainty"] <= t]
        rejected = len(df) - len(selected)
        coverage = len(selected) / len(df)

        if len(selected) > 0:
            sel_acc = (selected["Label"] == selected["Prediction"]).mean()
            improvement = (sel_acc - baseline_acc) * 100
        else:
            sel_acc = 0
            improvement = 0

        if sel_acc >= 0.95 and coverage > best_coverage_95:
            best_coverage_95 = coverage
            best_thresh_95 = t

        print("{:<14.6f} {:<10} {:<10} {:<12.2%} {:<12.2f} {:+.2f}%".format(
            t, len(selected), rejected, coverage, sel_acc * 100, improvement
        ))

    # --- Final report ---
    print()
    print("=" * 65)
    print("FINAL REPORT")
    print("=" * 65)

    print()
    print("MC Dropout Accuracy : {:.2f}%".format(baseline_acc * 100))

    if best_thresh_95 is not None:
        selected = df[df["Uncertainty"] <= best_thresh_95]
        coverage = len(selected) / len(df)
        sel_acc = (selected["Label"] == selected["Prediction"]).mean()

        print()
        print(">> 95%+ SELECTIVE ACCURACY ACHIEVED!")
        print("   Best Threshold    : {:.6f}".format(best_thresh_95))
        print("   Coverage          : {:.2%} ({}/{})".format(
            coverage, len(selected), len(df)
        ))
        print("   Selective Accuracy: {:.2f}%".format(sel_acc * 100))
        print("   Samples Rejected  : {} (referred to expert)".format(
            len(df) - len(selected)
        ))
    else:
        print()
        print(">> 95% selective accuracy NOT achieved at any threshold.")
        print("   Finding best available threshold...")

        best_t = None
        best_score = -1

        for t in thresholds:
            selected = df[df["Uncertainty"] <= t]
            if len(selected) > 10:
                sel_acc = (selected["Label"] == selected["Prediction"]).mean()
                coverage = len(selected) / len(df)
                score = sel_acc * 0.7 + coverage * 0.3

                if score > best_score:
                    best_score = score
                    best_t = t

        if best_t is not None:
            selected = df[df["Uncertainty"] <= best_t]
            coverage = len(selected) / len(df)
            sel_acc = (selected["Label"] == selected["Prediction"]).mean()

            print("   Best Threshold    : {:.6f}".format(best_t))
            print("   Coverage          : {:.2%}".format(coverage))
            print("   Selective Accuracy: {:.2f}%".format(sel_acc * 100))


if __name__ == "__main__":
    analyze(CSV_PATH)

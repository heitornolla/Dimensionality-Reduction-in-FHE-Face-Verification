import pandas as pd
import numpy as np
import glob
import os
from scipy import stats

FILE_MAPPING = {
    "ae_results.csv": "Autoencoder",
    "grp_results.csv": "Gaussian RP",
    "jl_results.csv": "JL (Hadamard)",
    "pca_results.csv": "PCA",
    "rsvd_results.csv": "RSVD",
    "srp_results.csv": "Sparse RP",
}


def load_data():
    all_dfs = []
    for f in glob.glob("results/*.csv"):
        name = os.path.basename(f)
        if name in FILE_MAPPING:
            df = pd.read_csv(f)
            df["Method"] = FILE_MAPPING[name]
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def analyze_results():
    df = load_data()

    # Descriptive Stats (Mean, Std, CI)
    stats_df = (
        df.groupby(["Method", "dimension"])["accuracy(%)"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Calculate 95% Confidence Interval
    # CI = 1.96 * (std / sqrt(n))
    stats_df["ci95"] = 1.96 * (stats_df["std"] / np.sqrt(stats_df["count"]))

    print("\n=== Descriptive Statistics (Accuracy) ===")
    print(stats_df.sort_values(["dimension", "mean"], ascending=[False, False]))

    # Statistical Significance Test (Wilcoxon)
    # We compare everything against PCA
    target_dims = [32, 16]  # Focus on the interesting dimensions
    baseline_method = "PCA"

    print(f"\n=== Statistical Significance (vs {baseline_method}) ===")
    for dim in target_dims:
        print(f"\nDimension {dim}:")

        base_samples = df[(df["Method"] == baseline_method) & (df["dimension"] == dim)][
            "accuracy(%)"
        ]

        if len(base_samples) < 2:
            print("  Not enough data for significance testing.")
            continue

        for method in FILE_MAPPING.values():
            if method == baseline_method:
                continue

            comp_samples = df[(df["Method"] == method) & (df["dimension"] == dim)][
                "accuracy(%)"
            ]

            if len(comp_samples) >= 2:
                # Perform Wilcoxon Rank-Sum Test (Mann-Whitney U)
                # We use 'two-sided' to check if they are significantly different
                stat, p_val = stats.mannwhitneyu(
                    base_samples, comp_samples, alternative="two-sided"
                )

                # Interpret p-value
                sig = "**" if p_val < 0.001 else "*" if p_val < 0.05 else "ns"
                print(f"  vs {method:15s}: p-value={p_val:.5f} ({sig})")
            else:
                print(f"  vs {method:15s}: No data")


if __name__ == "__main__":
    analyze_results()

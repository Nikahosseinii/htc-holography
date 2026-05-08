import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="slm_pseudo_hardware_stress_test/slm_stress_metrics.csv",
        help="Path to slm_stress_metrics.csv",
    )
    parser.add_argument(
        "--out_prefix",
        default="slm_pseudo_hardware_stress_test/slm_stress_one_figure",
        help="Output prefix for png/pdf/svg",
    )
    parser.add_argument(
        "--include_ideal",
        action="store_true",
        help="Include the trivial ideal point in the plot",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Optional: remove the ideal case because it is trivial
    if not args.include_ideal:
        df = df[df["case"] != "ideal"].copy()

    # Make labels nicer for the x-axis
    label_map = {
        "ideal": "ideal",
        "nonuniform_illumination": "nonuniform\nillumination",
        "wavefront_error": "wavefront\nerror",
        "pixel_crosstalk": "pixel\ncrosstalk",
        "misalignment": "misalignment",
        "all_uncalibrated": "all\nuncalibrated",
    }
    df["label"] = df["case"].map(label_map).fillna(df["case"])

    x = range(len(df))

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })

    fig, ax1 = plt.subplots(figsize=(10, 5.8))

    # Left axis: NRMSE
    line1 = ax1.plot(
        x,
        df["nrmse_vs_ideal"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="NRMSE vs. ideal replay",
    )
    ax1.set_xlabel("Simulated hardware effect")
    ax1.set_ylabel("NRMSE vs. ideal replay")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["label"])
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Right axis: Correlation
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        x,
        df["correlation_vs_ideal"],
        marker="s",
        linewidth=2.5,
        markersize=7,
        linestyle="--",
        label="Correlation vs. ideal replay",
    )
    ax2.set_ylabel("Correlation vs. ideal replay")
    ax2.set_ylim(bottom=min(0.0, df["correlation_vs_ideal"].min() - 0.05), top=1.05)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    plt.title("Pseudo-hardware stress test of SLM phase map")
    plt.tight_layout()

    out_prefix = Path(args.out_prefix)
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_prefix}.svg", bbox_inches="tight")

    print(f"Saved {out_prefix}.png")
    print(f"Saved {out_prefix}.pdf")
    print(f"Saved {out_prefix}.svg")


if __name__ == "__main__":
    main()

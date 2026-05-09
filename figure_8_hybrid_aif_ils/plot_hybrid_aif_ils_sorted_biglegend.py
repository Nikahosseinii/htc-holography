import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hybrid_aif_ils.csv")
    ap.add_argument("--out_prefix", default="commag_fig_hybrid_aif_ils_sorted_biglegend_inside")
    ap.add_argument("--smooth", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["id", "step", "model", "ils"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["ils"] = pd.to_numeric(df["ils"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["id", "step", "ils"])

    # Sort objects by median ILS across steps
    obj_order = (
        df.groupby("id")["ils"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    # Build one curve per step
    steps = sorted(df["step"].unique().tolist())
    x = np.arange(len(obj_order))

    plt.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    })

    style_cycle = [
        {"linestyle": "-",  "marker": "o"},
        {"linestyle": "--", "marker": "s"},
        {"linestyle": ":",  "marker": "^"},
        {"linestyle": "-.", "marker": "D"},
        {"linestyle": (0, (8, 2)), "marker": "v"},
        {"linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
        {"linestyle": (0, (1, 1)), "marker": "X"},
        {"linestyle": (0, (5, 1)), "marker": "*"},
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))

    for i, step in enumerate(steps):
        sub = df[df["step"] == step].groupby("id")["ils"].median()
        y = sub.reindex(obj_order).to_numpy()

        if args.smooth and args.smooth > 1:
            y = (
                pd.Series(y)
                .rolling(args.smooth, min_periods=1, center=True)
                .median()
                .to_numpy()
            )

        st = style_cycle[i % len(style_cycle)]

        ax.plot(
            x,
            y,
            linewidth=2.5,
            linestyle=st["linestyle"],
            marker=st["marker"],
            markersize=6,
            markevery=max(1, len(x) // 10),
            label=f"step={int(step)}"
        )

    ax.set_xlabel("Object rank (sorted by median hybrid AIF ILS)")
    ax.set_ylabel("ILS (CLIP-based)")
    ax.set_title("Hybrid AIF instruction locality across objects and interaction steps")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

    # Legend inside the plot
    ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=15,
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        edgecolor="none",
        markerscale=1.3,
        handlelength=2.8,
        labelspacing=0.45,
        columnspacing=1.1,
        handletextpad=0.6,
        borderpad=0.4
    )

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.out_prefix}.svg", bbox_inches="tight")
    plt.savefig(f"{args.out_prefix}.png", dpi=300, bbox_inches="tight")

    print(f"Saved {args.out_prefix}.pdf/.svg/.png")


if __name__ == "__main__":
    main()

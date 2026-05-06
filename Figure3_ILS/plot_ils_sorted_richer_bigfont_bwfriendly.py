import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ils_alpha_sweep/ils_alpha_sweep.csv",
                    help="Merged sweep CSV")
    ap.add_argument("--models", default="baseline,factorized_fixed",
                    help="Comma-separated models to plot")
    ap.add_argument("--alphas", default="0,0.2,0.5,1.0",
                    help="Comma-separated alpha values to plot")
    ap.add_argument("--sort_alpha", type=float, default=0.5,
                    help="Alpha used to define the sorting order (baseline at this alpha)")
    ap.add_argument("--out_prefix", default="commag_fig_sorted_richer_bigfont_bwfriendly",
                    help="Output prefix for pdf/svg")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Rolling window for smoothing (0 = off)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic cleanup
    df["alpha_preserve"] = pd.to_numeric(df["alpha_preserve"], errors="coerce")
    df["ils"] = pd.to_numeric(df["ils"], errors="coerce")
    df = df.dropna(subset=["id", "model", "alpha_preserve", "ils"])

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    alphas = parse_list(args.alphas)

    # Keep only requested models and alphas
    tol = 1e-9
    df = df[df["model"].isin(models)].copy()
    df = df[df["alpha_preserve"].apply(lambda a: any(abs(a - x) < tol for x in alphas))].copy()

    # Define sorting order using BASELINE at sort_alpha
    base = df[(df["model"] == "baseline") & (abs(df["alpha_preserve"] - args.sort_alpha) < tol)].copy()
    if base.empty:
        raise ValueError(
            f"No rows for baseline at alpha={args.sort_alpha}. "
            f"Check available alpha values in the CSV."
        )

    # For each id, take median ILS
    base_per_id = base.groupby("id")["ils"].median().sort_values()
    order = list(base_per_id.index)

    # Helper: get series aligned to the order for a (model, alpha)
    def series_for(model: str, alpha: float):
        sub = df[(df["model"] == model) & (abs(df["alpha_preserve"] - alpha) < tol)]
        per_id = sub.groupby("id")["ils"].median()
        y = per_id.reindex(order).to_numpy()
        if args.smooth and args.smooth > 1:
            y = pd.Series(y).rolling(args.smooth, min_periods=1, center=True).median().to_numpy()
        return y

    x = np.arange(len(order))

    # Bigger readable fonts
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    # Color + linestyle + marker combinations
    # Distinguishable in color and also in grayscale
    style_map = {
        0.0: {"color": "tab:blue",   "linestyle": "-",  "marker": "o"},
        0.2: {"color": "tab:orange", "linestyle": "--", "marker": "s"},
        0.5: {"color": "tab:green",  "linestyle": ":",  "marker": "^"},
        1.0: {"color": "tab:red",    "linestyle": "-.", "marker": "D"},
    }

    fallback_styles = [
        {"color": "tab:purple", "linestyle": (0, (8, 2)),     "marker": "v"},
        {"color": "tab:brown",  "linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
        {"color": "tab:pink",   "linestyle": (0, (1, 1)),     "marker": "X"},
        {"color": "tab:gray",   "linestyle": (0, (5, 1)),     "marker": "*"},
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for ax, model in zip(axes, ["baseline", "factorized_fixed"]):
        for idx, a in enumerate(alphas):
            y = series_for(model, a)

            style = style_map.get(a)
            if style is None:
                style = fallback_styles[idx % len(fallback_styles)]

            ax.plot(
                x,
                y,
                linewidth=2.5,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(x) // 12),
                markersize=5.5,
                label=f"α={a:g}"
            )

        ax.set_ylabel("ILS (CLIP-based)", fontsize=16)
        ax.set_title(model.replace("_", "-"), fontsize=17)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        ax.legend(frameon=False, ncol=min(4, len(alphas)), fontsize=12)

    axes[-1].set_xlabel(
        f"Sample position (sorted by baseline ILS at α={args.sort_alpha:g})",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.out_prefix}.svg", bbox_inches="tight")
    plt.savefig(f"{args.out_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved {args.out_prefix}.pdf/.svg")


if __name__ == "__main__":
    main()

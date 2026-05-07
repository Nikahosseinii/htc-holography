import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_profile(path: Path) -> dict:
    d = json.loads(path.read_text())

    if "summary" not in d:
        raise KeyError(f"{path} missing 'summary' field")

    s = d["summary"]

    required = [
        "L_proc_s_mean",
        "L_enc_s_mean",
        "L_dec_s_mean",
        "payload_bits_mean",
    ]

    for k in required:
        if k not in s:
            raise KeyError(f"{path} missing key: summary['{k}']")

    return s


def il_end_to_end_seconds(
    prof: dict,
    R_bps: float,
    rho: float,
    L_sense_s: float,
    L_render_s: float,
    L_display_s: float,
) -> float:
    B_bits = float(prof["payload_bits_mean"])
    if B_bits <= 0:
        raise ValueError("payload_bits_mean must be > 0")

    L_tx = B_bits / R_bps

    if rho <= 0:
        L_queue = 0.0
    else:
        if rho >= 1.0:
            return float("inf")
        mu = R_bps / B_bits
        L_queue = rho / (mu * (1.0 - rho))

    L_enc = float(prof["L_enc_s_mean"])
    L_dec = float(prof["L_dec_s_mean"])
    L_proc = float(prof["L_proc_s_mean"])

    return L_sense_s + L_enc + L_queue + L_tx + L_dec + L_proc + L_render_s + L_display_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", required=True, help="JSON profile files")
    ap.add_argument("--labels", nargs="+", required=True, help="Labels matching profiles")
    ap.add_argument("--out", default="IL_full_bigfont.png")

    ap.add_argument("--R_mbps", nargs="+", type=float, default=[50, 100, 300, 1000, 3000, 10000])
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.8])

    ap.add_argument("--L_sense_ms", type=float, default=5.0)
    ap.add_argument("--L_render_ms", type=float, default=5.0)
    ap.add_argument("--L_display_ms", type=float, default=16.7)

    args = ap.parse_args()

    prof_paths = [Path(p) for p in args.profiles]
    if len(prof_paths) != len(args.labels):
        raise ValueError("profiles and labels must have same length")

    profs = [load_profile(p) for p in prof_paths]

    R_list_mbps = args.R_mbps
    R_list_bps = [r * 1e6 for r in R_list_mbps]
    rhos = args.rhos

    L_sense_s = args.L_sense_ms / 1000.0
    L_render_s = args.L_render_ms / 1000.0
    L_display_s = args.L_display_ms / 1000.0

    # Bigger global fonts
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })

    fig, ax = plt.subplots(figsize=(12, 7.5))

    model_colors = {
        "Baseline (50 DDIM)": "tab:blue",
        "Factorized (25 DDIM)": "tab:orange",
        "Factorized (10 DDIM)": "tab:green",
    }

    rho_styles = {
        0.0: "-",
        0.3: "--",
        0.6: "-.",
        0.8: ":",
    }

    for prof, label in zip(profs, args.labels):
        for rho in rhos:
            ys_ms = []
            for R_bps in R_list_bps:
                il_s = il_end_to_end_seconds(
                    prof, R_bps, rho, L_sense_s, L_render_s, L_display_s
                )
                ys_ms.append(il_s * 1000.0)

            ax.plot(
                R_list_mbps,
                ys_ms,
                marker="o",
                markersize=7,
                linewidth=2.2,
                color=model_colors.get(label, None),
                linestyle=rho_styles[rho],
                label=f"{label}, ρ={rho}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Link rate R (Mbps, log scale)")
    ax.set_ylabel("End-to-end interaction latency IL (ms)")
    ax.set_title("Interaction Latency vs Link Rate (Mon3tr-style E2E model)", pad=12)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="both", which="major", length=6, width=1.2)

    ax.legend(ncols=2, frameon=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()

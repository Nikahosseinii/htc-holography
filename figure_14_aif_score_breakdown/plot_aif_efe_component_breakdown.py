#!/usr/bin/env python3
"""
plot_aif_efe_component_breakdown.py

Creates an IWAI-friendly interpretability figure for the Hybrid AIF policy.

Input:
    outputs_compare_factorized_vs_hybrid_regret_tuned_v3/compare_metrics.csv

Output:
    outputs_aif_efe_breakdown/aif_efe_component_breakdown.png
    outputs_aif_efe_breakdown/aif_efe_component_breakdown.pdf
    outputs_aif_efe_breakdown/aif_efe_component_breakdown.svg
    outputs_aif_efe_breakdown/aif_efe_component_breakdown_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Z_SHARED_BYTES = 946_176
BITS_SCALE = float(Z_SHARED_BYTES * 8)

REF_ERR_SCALE = 0.05
DRIFT_ERR_SCALE = 0.03
LAT_MS_SCALE = 4000.0

W_REF = 1.00
W_BITS = 0.35
W_LAT = 0.25
W_TEMP = 0.20


ACTION_LABELS = {
    "local_transform": "Local",
    "send_view": "View",
    "refresh_shared": "Refresh",
}


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def add_score_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "ref_error",
        "recon_error_vs_prev",
        "payload_bits",
        "proc_ms_total",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    out["risk_component"] = W_REF * clip01(out["ref_error"] / REF_ERR_SCALE)
    out["ambiguity_component"] = W_TEMP * clip01(
        out["recon_error_vs_prev"] / DRIFT_ERR_SCALE
    )
    out["bandwidth_component"] = W_BITS * clip01(out["payload_bits"] / BITS_SCALE)
    out["latency_component"] = W_LAT * clip01(out["proc_ms_total"] / LAT_MS_SCALE)

    out["efe_score"] = (
        out["risk_component"]
        + out["ambiguity_component"]
        + out["bandwidth_component"]
        + out["latency_component"]
    )

    return out


def majority_action(actions: pd.Series) -> str:
    if actions.empty:
        return "unknown"
    return actions.value_counts().idxmax()


def make_breakdown_figure(
    df: pd.DataFrame,
    out_dir: Path,
    obj_id: str | None = None,
    prefix: str = "aif_efe_component_breakdown",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hybrid = df[df["method"].astype(str).str.lower().eq("hybrid")].copy()

    if hybrid.empty:
        raise ValueError("No rows with method == 'hybrid' were found in the CSV.")

    if obj_id is not None:
        hybrid = hybrid[hybrid["obj_id"].astype(str).eq(str(obj_id))].copy()
        if hybrid.empty:
            raise ValueError(f"No Hybrid AIF rows found for obj_id={obj_id}")

    hybrid = add_score_components(hybrid)

    components = [
        "risk_component",
        "ambiguity_component",
        "bandwidth_component",
        "latency_component",
    ]

    component_labels = [
        "Risk\n(ref. error)",
        "Ambiguity\n(drift)",
        "Bandwidth\n(payload)",
        "Latency\n(proc.)",
    ]

    step_df = (
        hybrid.groupby("step", as_index=False)[components + ["efe_score"]]
        .mean()
        .sort_values("step")
    )

    action_major = (
        hybrid.groupby("step")["action"]
        .apply(majority_action)
        .reset_index(name="majority_action")
        .sort_values("step")
    )

    action_counts = hybrid.groupby(["step", "action"]).size().reset_index(name="count")
    action_counts["fraction"] = action_counts.groupby("step")["count"].transform(
        lambda x: x / x.sum()
    )

    steps = step_df["step"].to_numpy()
    x = np.arange(len(steps))

    fig = plt.figure(figsize=(8.0, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.05], hspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])

    bottom = np.zeros(len(step_df))
    hatches = ["", "///", "\\\\\\", "..."]

    for comp, label, hatch in zip(components, component_labels, hatches):
        values = step_df[comp].to_numpy()
        ax1.bar(
            x,
            values,
            bottom=bottom,
            label=label,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
        )
        bottom += values

    ax1.plot(
        x,
        step_df["efe_score"].to_numpy(),
        marker="D",
        linewidth=1.4,
        label="Total score",
    )

    ax1.set_ylabel("Weighted normalized score")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in steps])
    ax1.set_xlabel("Interaction step")
    ax1.set_title("(a) Expected-free-energy-inspired score components")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(ncol=3, fontsize=8, frameon=True)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    action_order = ["local_transform", "send_view", "refresh_shared"]
    action_hatches = {
        "local_transform": "",
        "send_view": "///",
        "refresh_shared": "\\\\\\",
    }

    bottom = np.zeros(len(steps))

    for action in action_order:
        vals = []
        for s in steps:
            row = action_counts[
                (action_counts["step"] == s) & (action_counts["action"] == action)
            ]
            vals.append(float(row["fraction"].iloc[0]) if len(row) else 0.0)

        ax2.bar(
            x,
            vals,
            bottom=bottom,
            edgecolor="black",
            linewidth=0.6,
            hatch=action_hatches[action],
            label=ACTION_LABELS[action],
        )
        bottom += np.array(vals)

    for i, s in enumerate(steps):
        action = action_major.loc[action_major["step"] == s, "majority_action"].iloc[0]
        label = ACTION_LABELS.get(action, action)
        ax2.text(i, 1.04, label, ha="center", va="bottom", fontsize=8)

    ax2.set_ylim(0, 1.22)
    ax2.set_ylabel("Action\nfraction")
    ax2.set_xlabel("Interaction step")
    ax2.set_title("(b) Selected Hybrid AIF actions")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in steps])
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(ncol=3, fontsize=8, loc="upper right", frameon=True)

    fig.suptitle("Interpreting Hybrid AIF Update Decisions", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    suffix = f"_{obj_id}" if obj_id is not None else ""

    png_path = out_dir / f"{prefix}{suffix}.png"
    pdf_path = out_dir / f"{prefix}{suffix}.pdf"
    svg_path = out_dir / f"{prefix}{suffix}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary = step_df.merge(action_major, on="step", how="left")
    summary_path = out_dir / f"{prefix}{suffix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="outputs_compare_factorized_vs_hybrid_regret_tuned_v3/compare_metrics.csv",
        help="Path to compare_metrics.csv generated by the Hybrid AIF rollout script.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_aif_efe_breakdown",
        help="Directory where the figure will be saved.",
    )

    parser.add_argument(
        "--obj_id",
        type=str,
        default=None,
        help="Optional object id. If omitted, the figure averages over all objects.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run the Hybrid AIF comparison script first."
        )

    df = pd.read_csv(csv_path)

    make_breakdown_figure(
        df=df,
        out_dir=Path(args.out_dir),
        obj_id=args.obj_id,
    )


if __name__ == "__main__":
    main()

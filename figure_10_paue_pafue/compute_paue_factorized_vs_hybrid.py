import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def infer_object_col(df: pd.DataFrame) -> str:
    for col in ["obj_id", "id", "object_id"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find object column. Expected one of: obj_id, id, object_id")


def infer_error_col(df: pd.DataFrame) -> str:
    for col in ["ref_error", "error_vs_ref", "mse_vs_ref", "recon_error_ref"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find reference-error column. Expected one of: "
        "ref_error, error_vs_ref, mse_vs_ref, recon_error_ref"
    )


def clean_method_name(name: str) -> str:
    name = str(name).lower().strip()

    if "hybrid" in name or "aif" in name:
        return "Hybrid AIF"

    if "factorized" in name or "fixed" in name:
        return "Fixed Factorized"

    return str(name)


def compute_paue_for_horizon(
    df: pd.DataFrame,
    object_col: str,
    method_col: str,
    step_col: str,
    error_col: str,
    horizon: int,
) -> pd.DataFrame:
    rows = []

    for (obj_id, method), group in df.groupby([object_col, method_col]):
        group = group.sort_values(step_col).copy()

        step_to_error = {
            int(row[step_col]): float(row[error_col])
            for _, row in group.iterrows()
        }

        steps = sorted(step_to_error.keys())

        for t in steps:
            future_steps = [t + k for k in range(1, horizon + 1)]

            if not all(s in step_to_error for s in future_steps):
                continue

            future_errors = [step_to_error[s] for s in future_steps]

            paue = float(np.mean(future_errors))
            pafue = float(future_errors[-1])

            rows.append({
                "id": obj_id,
                "method": method,
                "start_step": t,
                "horizon": horizon,
                "paue": paue,
                "pafue": pafue,
            })

    return pd.DataFrame(rows)


def summarize(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        raise ValueError("No valid PAUE/PAFUE rows were computed. Check step continuity and horizon values.")

    summary = (
        metric_df
        .groupby(["method", "horizon"])
        .agg(
            paue_mean=("paue", "mean"),
            paue_std=("paue", "std"),
            pafue_mean=("pafue", "mean"),
            pafue_std=("pafue", "std"),
            n=("paue", "count"),
        )
        .reset_index()
    )

    summary["paue_sem"] = summary["paue_std"] / np.sqrt(summary["n"])
    summary["pafue_sem"] = summary["pafue_std"] / np.sqrt(summary["n"])

    return summary


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.2))

    for method in sorted(summary["method"].unique()):
        sub = summary[summary["method"] == method].sort_values("horizon")

        y_col = f"{metric}_mean"
        err_col = f"{metric}_sem"

        plt.errorbar(
            sub["horizon"],
            sub[y_col],
            yerr=sub[err_col],
            marker="o",
            linewidth=2.4,
            markersize=7,
            capsize=4,
            label=method,
        )

    plt.xlabel("Future horizon $H$ (interaction steps)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title("Prediction-aware update reliability under viewpoint motion", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=13, frameon=True)
    plt.tight_layout()

    plt.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")

    print(f"Saved {out_path.with_suffix('.png')}")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    print(f"Saved {out_path.with_suffix('.svg')}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="outputs_compare_factorized_vs_hybrid/compare_metrics.csv",
        help="Path to compare_metrics.csv produced by the factorized-vs-hybrid comparison code.",
    )

    parser.add_argument(
        "--out_dir",
        default="metrics_paue_factorized_vs_hybrid",
        help="Directory to save PAUE/PAFUE CSVs and figures.",
    )

    parser.add_argument(
        "--max_horizon",
        type=int,
        default=4,
        help="Maximum future horizon H to evaluate.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    object_col = infer_object_col(df)
    error_col = infer_error_col(df)

    required = [object_col, "method", "step", error_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["method"] = df["method"].apply(clean_method_name)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df[error_col] = pd.to_numeric(df[error_col], errors="coerce")

    df = df.dropna(subset=[object_col, "method", "step", error_col])
    df["step"] = df["step"].astype(int)

    all_metric_rows = []

    for horizon in range(1, args.max_horizon + 1):
        h_df = compute_paue_for_horizon(
            df=df,
            object_col=object_col,
            method_col="method",
            step_col="step",
            error_col=error_col,
            horizon=horizon,
        )
        all_metric_rows.append(h_df)

    metric_df = pd.concat(all_metric_rows, ignore_index=True)
    summary_df = summarize(metric_df)

    metric_csv = out_dir / "paue_pafue_per_window.csv"
    summary_csv = out_dir / "paue_pafue_summary.csv"

    metric_df.to_csv(metric_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved {metric_csv}")
    print(f"Saved {summary_csv}")

    print("\n=== PAUE/PAFUE SUMMARY ===")
    print(summary_df.to_string(index=False))

    plot_metric(
        summary=summary_df,
        metric="paue",
        ylabel="Mean PAUE ↓",
        out_path=out_dir / "paue_vs_horizon",
    )

    plot_metric(
        summary=summary_df,
        metric="pafue",
        ylabel="Mean PAFUE ↓",
        out_path=out_dir / "pafue_vs_horizon",
    )


if __name__ == "__main__":
    main()

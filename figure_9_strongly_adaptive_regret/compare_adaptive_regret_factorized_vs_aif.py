#!/usr/bin/env python3
"""
compare_adaptive_regret_factorized_vs_aif.py

Compute strongly adaptive regret for:
- factorized MVDream
- hybrid AIF

using the interval-based definition from adaptive-regret literature,
instantiated over method-level losses from compare_metrics.csv.

Comparator set:
    Omega = {factorized, hybrid}

For each object and interval length tau:
    SA-Regret(method, tau) =
        max over contiguous intervals I of length tau:
            cumulative_loss(method on I) - min_{m in Omega} cumulative_loss(m on I)

This script assumes your CSV was produced by the comparison rollout code and
contains per-step rows with at least these fields:
    obj_id, method, step, payload_bits, proc_ms_total, ref_error, recon_error_vs_prev

Recommended usage:
    python compare_adaptive_regret_factorized_vs_aif.py \
      --csv outputs_compare_factorized_vs_hybrid/compare_metrics.csv \
      --taus 2 3 4 5 6 8 \
      --out_json adaptive_regret_results.json \
      --plot adaptive_regret_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


REQUIRED_COLUMNS = {
    "obj_id",
    "method",
    "step",
    "payload_bits",
    "proc_ms_total",
    "ref_error",
    "recon_error_vs_prev",
}

VALID_METHODS = {"factorized", "hybrid"}


@dataclass
class StepRecord:
    obj_id: str
    method: str
    step: int
    payload_bits: float
    proc_ms_total: float
    ref_error: float
    recon_error_vs_prev: float
    loss: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strongly adaptive regret of factorized MVDream vs hybrid AIF."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to compare_metrics.csv",
    )
    parser.add_argument(
        "--taus",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 8],
        help="Interval lengths tau for strongly adaptive regret.",
    )
    parser.add_argument(
        "--w_ref",
        type=float,
        default=1.0,
        help="Weight for reference error term.",
    )
    parser.add_argument(
        "--w_bits",
        type=float,
        default=0.35,
        help="Weight for payload bits term.",
    )
    parser.add_argument(
        "--w_lat",
        type=float,
        default=0.25,
        help="Weight for processing latency term.",
    )
    parser.add_argument(
        "--w_temp",
        type=float,
        default=0.20,
        help="Weight for temporal instability term (recon_error_vs_prev).",
    )
    parser.add_argument(
        "--norm_scope",
        choices=["global", "per_object"],
        default="per_object",
        help="Normalize loss terms globally or per object.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="adaptive_regret_results.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="Optional output plot path.",
    )
    return parser.parse_args()


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def safe_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def load_csv(path: str) -> List[StepRecord]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        rows: List[StepRecord] = []
        for row in reader:
            method = str(row["method"]).strip()
            if method not in VALID_METHODS:
                continue

            rows.append(
                StepRecord(
                    obj_id=str(row["obj_id"]).strip(),
                    method=method,
                    step=safe_int(row["step"]),
                    payload_bits=safe_float(row["payload_bits"]),
                    proc_ms_total=safe_float(row["proc_ms_total"]),
                    ref_error=safe_float(row["ref_error"]),
                    recon_error_vs_prev=safe_float(row["recon_error_vs_prev"]),
                )
            )
    if not rows:
        raise RuntimeError("No valid factorized/hybrid rows found in CSV.")
    return rows


def minmax_norm(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isclose(vmax, vmin):
        return [0.0 for _ in values]
    return ((arr - vmin) / (vmax - vmin)).tolist()


def assign_losses(
    rows: List[StepRecord],
    w_ref: float,
    w_bits: float,
    w_lat: float,
    w_temp: float,
    norm_scope: str,
) -> None:
    if norm_scope == "global":
        refs = [r.ref_error for r in rows]
        bits = [r.payload_bits for r in rows]
        lats = [r.proc_ms_total for r in rows]
        temps = [r.recon_error_vs_prev for r in rows]

        ref_n = minmax_norm(refs)
        bits_n = minmax_norm(bits)
        lats_n = minmax_norm(lats)
        temps_n = minmax_norm(temps)

        for r, a, b, c, d in zip(rows, ref_n, bits_n, lats_n, temps_n):
            r.loss = w_ref * a + w_bits * b + w_lat * c + w_temp * d
        return

    grouped: Dict[str, List[StepRecord]] = defaultdict(list)
    for r in rows:
        grouped[r.obj_id].append(r)

    for obj_id, grp in grouped.items():
        refs = [r.ref_error for r in grp]
        bits = [r.payload_bits for r in grp]
        lats = [r.proc_ms_total for r in grp]
        temps = [r.recon_error_vs_prev for r in grp]

        ref_n = minmax_norm(refs)
        bits_n = minmax_norm(bits)
        lats_n = minmax_norm(lats)
        temps_n = minmax_norm(temps)

        for r, a, b, c, d in zip(grp, ref_n, bits_n, lats_n, temps_n):
            r.loss = w_ref * a + w_bits * b + w_lat * c + w_temp * d


def build_aligned_sequences(
    rows: List[StepRecord],
) -> Dict[str, Dict[str, List[StepRecord]]]:
    grouped: Dict[str, Dict[str, Dict[int, StepRecord]]] = defaultdict(lambda: defaultdict(dict))

    for r in rows:
        grouped[r.obj_id][r.method][r.step] = r

    out: Dict[str, Dict[str, List[StepRecord]]] = {}
    for obj_id, method_map in grouped.items():
        if "factorized" not in method_map or "hybrid" not in method_map:
            continue

        common_steps = sorted(set(method_map["factorized"].keys()) & set(method_map["hybrid"].keys()))
        if not common_steps:
            continue

        out[obj_id] = {
            "factorized": [method_map["factorized"][s] for s in common_steps],
            "hybrid": [method_map["hybrid"][s] for s in common_steps],
        }

    if not out:
        raise RuntimeError(
            "No objects have aligned factorized and hybrid step sequences. "
            "Check that both methods were logged on the same object IDs and steps."
        )

    return out


def interval_sum(prefix: np.ndarray, start: int, end_exclusive: int) -> float:
    return float(prefix[end_exclusive] - prefix[start])


def strongly_adaptive_regret_for_tau(
    factor_losses: List[float],
    hybrid_losses: List[float],
    tau: int,
) -> Dict[str, float]:
    n = len(factor_losses)
    if tau <= 0 or tau > n:
        return {"factorized": float("nan"), "hybrid": float("nan")}

    f = np.asarray(factor_losses, dtype=np.float64)
    h = np.asarray(hybrid_losses, dtype=np.float64)

    pf = np.concatenate([[0.0], np.cumsum(f)])
    ph = np.concatenate([[0.0], np.cumsum(h)])

    worst_factor = -float("inf")
    worst_hybrid = -float("inf")

    for start in range(0, n - tau + 1):
        end = start + tau

        lf = interval_sum(pf, start, end)
        lh = interval_sum(ph, start, end)

        oracle = min(lf, lh)

        worst_factor = max(worst_factor, lf - oracle)
        worst_hybrid = max(worst_hybrid, lh - oracle)

    return {
        "factorized": float(worst_factor),
        "hybrid": float(worst_hybrid),
    }


def summarize_by_object_and_tau(
    aligned: Dict[str, Dict[str, List[StepRecord]]],
    taus: List[int],
) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], Dict[int, Dict[str, Dict[str, float]]]]:
    per_object: Dict[str, Dict[int, Dict[str, float]]] = {}
    aggregate: Dict[int, Dict[str, Dict[str, float]]] = {}

    for obj_id, seqs in aligned.items():
        factor_losses = [r.loss for r in seqs["factorized"]]
        hybrid_losses = [r.loss for r in seqs["hybrid"]]
        per_object[obj_id] = {}

        for tau in taus:
            vals = strongly_adaptive_regret_for_tau(factor_losses, hybrid_losses, tau)
            per_object[obj_id][tau] = vals

    for tau in taus:
        fvals = []
        hvals = []
        for obj_id in per_object:
            vals = per_object[obj_id][tau]
            if not math.isnan(vals["factorized"]):
                fvals.append(vals["factorized"])
            if not math.isnan(vals["hybrid"]):
                hvals.append(vals["hybrid"])

        aggregate[tau] = {
            "factorized": {
                "mean": float(np.mean(fvals)) if fvals else float("nan"),
                "std": float(np.std(fvals)) if fvals else float("nan"),
            },
            "hybrid": {
                "mean": float(np.mean(hvals)) if hvals else float("nan"),
                "std": float(np.std(hvals)) if hvals else float("nan"),
            },
        }

    return per_object, aggregate


def save_json(
    path: str,
    args: argparse.Namespace,
    aligned: Dict[str, Dict[str, List[StepRecord]]],
    per_object: Dict[str, Dict[int, Dict[str, float]]],
    aggregate: Dict[int, Dict[str, Dict[str, float]]],
) -> None:
    out = {
        "definition": {
            "name": "method-level strongly adaptive regret",
            "description": (
                "Maximum interval-wise regret over intervals of length tau, "
                "with comparator equal to the best fixed method in hindsight "
                "among {factorized, hybrid} on that interval."
            ),
        },
        "loss_definition": {
            "loss_t": (
                f"{args.w_ref} * norm(ref_error) + "
                f"{args.w_bits} * norm(payload_bits) + "
                f"{args.w_lat} * norm(proc_ms_total) + "
                f"{args.w_temp} * norm(recon_error_vs_prev)"
            ),
            "normalization_scope": args.norm_scope,
        },
        "num_objects": len(aligned),
        "objects": sorted(aligned.keys()),
        "taus": args.taus,
        "aggregate": aggregate,
        "per_object": {
            obj_id: {str(tau): vals for tau, vals in tau_map.items()}
            for obj_id, tau_map in per_object.items()
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def maybe_plot(path: str, aggregate: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    if not path:
        return
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not available; skipping plot.")
        return

    taus = sorted(aggregate.keys())
    f_means = [aggregate[t]["factorized"]["mean"] for t in taus]
    h_means = [aggregate[t]["hybrid"]["mean"] for t in taus]
    f_stds = [aggregate[t]["factorized"]["std"] for t in taus]
    h_stds = [aggregate[t]["hybrid"]["std"] for t in taus]

    plt.figure(figsize=(8, 5.5))
    plt.plot(taus, f_means, marker="o", linewidth=2.0, label="Factorized")
    plt.plot(taus, h_means, marker="o", linewidth=2.0, label="Hybrid AIF")
    plt.fill_between(taus, np.array(f_means) - np.array(f_stds), np.array(f_means) + np.array(f_stds), alpha=0.18)
    plt.fill_between(taus, np.array(h_means) - np.array(h_stds), np.array(h_means) + np.array(h_stds), alpha=0.18)

    plt.xlabel("Interval length τ")
    plt.ylabel("Strongly adaptive regret")
    plt.title("Strongly adaptive regret vs interval length")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary(aggregate: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    print("\n=== Strongly Adaptive Regret Summary ===")
    print("Lower is better.\n")
    for tau in sorted(aggregate.keys()):
        f_mean = aggregate[tau]["factorized"]["mean"]
        f_std = aggregate[tau]["factorized"]["std"]
        h_mean = aggregate[tau]["hybrid"]["mean"]
        h_std = aggregate[tau]["hybrid"]["std"]

        print(f"tau = {tau}")
        print(f"  Factorized : {f_mean:.6f} ± {f_std:.6f}")
        print(f"  Hybrid AIF : {h_mean:.6f} ± {h_std:.6f}")
        print(f"  Delta (Hybrid - Factorized): {h_mean - f_mean:+.6f}")
        print()


def main() -> None:
    args = parse_args()

    rows = load_csv(args.csv)
    assign_losses(
        rows=rows,
        w_ref=args.w_ref,
        w_bits=args.w_bits,
        w_lat=args.w_lat,
        w_temp=args.w_temp,
        norm_scope=args.norm_scope,
    )
    aligned = build_aligned_sequences(rows)
    per_object, aggregate = summarize_by_object_and_tau(aligned, args.taus)

    save_json(args.out_json, args, aligned, per_object, aggregate)
    maybe_plot(args.plot, aggregate)
    print_summary(aggregate)

    print(f"[OK] Saved JSON: {args.out_json}")
    if args.plot:
        print(f"[OK] Saved plot: {args.plot}")


if __name__ == "__main__":
    main()

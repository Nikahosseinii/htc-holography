#!/usr/bin/env python3
"""Create a richer one-panel SDOP figure using update action as a variable.

The script reads:
    outputs_sdop_aif_vs_factorized/sdop_per_event.csv

It plots:
    1. Fixed Factorized: all events
    2. Hybrid AIF: all events
    3. Fixed Factorized: view-update events
    4. Hybrid AIF: view-update events
    5. Hybrid AIF: shared-refresh events
    6. Hybrid AIF: local-transform events, when available

The overall method curves include 95% Wilson confidence bands.
Action-specific curves are shown only when enough samples exist.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ACTION_LABELS = {
    "local_transform": "Local transform",
    "send_view": "View update",
    "refresh_shared": "Shared refresh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a richer one-panel SDOP figure."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "outputs_sdop_aif_vs_factorized/sdop_per_event.csv"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_sdop_aif_vs_factorized"),
    )

    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--threshold-points",
        type=int,
        default=81,
    )

    parser.add_argument(
        "--min-events",
        type=int,
        default=3,
        help="Minimum samples needed for an action-specific curve.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if args.threshold_min < 0:
        parser.error("--threshold-min must be non-negative.")

    if args.threshold_max <= args.threshold_min:
        parser.error(
            "--threshold-max must be greater than --threshold-min."
        )

    if args.threshold_points < 2:
        parser.error("--threshold-points must be at least 2.")

    return args


def read_event_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Run compare_sdop_aif_vs_factorized.py first."
        )

    required_columns = {
        "method",
        "method_label",
        "semantic_distortion",
        "action",
    }

    rows: List[Dict[str, object]] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if (
            reader.fieldnames is None
            or not required_columns.issubset(reader.fieldnames)
        ):
            raise ValueError(
                f"The CSV must contain: "
                f"{sorted(required_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                distortion = float(
                    row["semantic_distortion"]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid distortion on line "
                    f"{line_number}"
                ) from exc

            if not math.isfinite(distortion):
                continue

            rows.append(
                {
                    "method": str(row["method"]).strip(),
                    "method_label": str(
                        row["method_label"]
                    ).strip(),
                    "action": str(
                        row.get("action", "")
                    ).strip(),
                    "semantic_distortion": distortion,
                }
            )

    if not rows:
        raise RuntimeError(
            "No valid rows were found in the input CSV."
        )

    return rows


def wilson_interval(
    successes: int,
    total: int,
) -> Tuple[float, float]:
    """Return a two-sided 95% Wilson confidence interval."""

    if total <= 0:
        return math.nan, math.nan

    z = 1.959963984540054
    probability = successes / total

    denominator = 1.0 + z**2 / total

    center = (
        probability
        + z**2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            probability
            * (1.0 - probability)
            / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def select_distortions(
    rows: Sequence[Dict[str, object]],
    method: str,
    action: str | None = None,
) -> np.ndarray:
    selected = [
        float(row["semantic_distortion"])
        for row in rows
        if row["method"] == method
        and (
            action is None
            or row["action"] == action
        )
    ]

    return np.asarray(
        selected,
        dtype=np.float64,
    )


def make_sdop_curve(
    distortions: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distortions.size == 0:
        raise ValueError(
            "Cannot create an SDOP curve from no samples."
        )

    sdop = np.zeros_like(
        thresholds,
        dtype=np.float64,
    )

    lower = np.zeros_like(
        thresholds,
        dtype=np.float64,
    )

    upper = np.zeros_like(
        thresholds,
        dtype=np.float64,
    )

    for index, threshold in enumerate(thresholds):
        outage_count = int(
            np.sum(distortions > threshold)
        )

        sdop[index] = (
            outage_count / distortions.size
        )

        lower[index], upper[index] = (
            wilson_interval(
                outage_count,
                distortions.size,
            )
        )

    return sdop, lower, upper


def sdop_at_threshold(
    distortions: np.ndarray,
    threshold: float,
) -> float:
    if distortions.size == 0:
        return math.nan

    return float(
        np.mean(distortions > threshold)
    )


def write_curve_csv(
    path: Path,
    series_list: Sequence[Dict[str, object]],
    thresholds: np.ndarray,
) -> None:
    output_rows: List[Dict[str, object]] = []

    for series in series_list:
        for threshold, sdop, low, high in zip(
            thresholds,
            series["sdop"],
            series["lower"],
            series["upper"],
        ):
            output_rows.append(
                {
                    "series_id": series["series_id"],
                    "series_label": series["label"],
                    "method": series["method"],
                    "action": (
                        series["action"]
                        if series["action"] is not None
                        else "all"
                    ),
                    "num_events": series["num_events"],
                    "threshold": float(threshold),
                    "sdop": float(sdop),
                    "sdop_wilson_95_low": float(low),
                    "sdop_wilson_95_high": float(high),
                }
            )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    args = parse_args()

    rows = read_event_rows(
        args.input_csv
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = np.linspace(
        args.threshold_min,
        args.threshold_max,
        args.threshold_points,
        dtype=np.float64,
    )

    series_specs: List[Dict[str, object]] = [
        {
            "series_id": "factorized_all",
            "method": "factorized",
            "action": None,
            "label_base": "Fixed Factorized",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.8,
            "confidence_band": True,
        },
        {
            "series_id": "hybrid_all",
            "method": "hybrid",
            "action": None,
            "label_base": "Hybrid AIF",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.8,
            "confidence_band": True,
        },
    ]

    # Add a curve for fixed factorized view updates.
    factorized_view = select_distortions(
        rows,
        method="factorized",
        action="send_view",
    )

    if factorized_view.size >= args.min_events:
        series_specs.append(
            {
                "series_id": "factorized_view",
                "method": "factorized",
                "action": "send_view",
                "label_base": (
                    "Fixed Factorized: view update"
                ),
                "linestyle": "-.",
                "marker": "^",
                "linewidth": 1.8,
                "confidence_band": False,
            }
        )

    # Add curves for all Hybrid-AIF actions that exist.
    hybrid_action_styles = [
        (
            "local_transform",
            ":",
            "D",
        ),
        (
            "send_view",
            (0, (5, 2)),
            "v",
        ),
        (
            "refresh_shared",
            (0, (1, 1)),
            "P",
        ),
    ]

    for action, linestyle, marker in hybrid_action_styles:
        values = select_distortions(
            rows,
            method="hybrid",
            action=action,
        )

        if values.size < args.min_events:
            continue

        series_specs.append(
            {
                "series_id": f"hybrid_{action}",
                "method": "hybrid",
                "action": action,
                "label_base": (
                    "Hybrid AIF: "
                    + ACTION_LABELS[action].lower()
                ),
                "linestyle": linestyle,
                "marker": marker,
                "linewidth": 1.8,
                "confidence_band": False,
            }
        )

    plotted_series: List[Dict[str, object]] = []

    for spec in series_specs:
        distortions = select_distortions(
            rows,
            method=str(spec["method"]),
            action=spec["action"],
        )

        if distortions.size == 0:
            continue

        sdop, lower, upper = make_sdop_curve(
            distortions,
            thresholds,
        )

        label = (
            f"{spec['label_base']} "
            f"($n={distortions.size}$)"
        )

        plotted_series.append(
            {
                **spec,
                "label": label,
                "num_events": int(
                    distortions.size
                ),
                "distortions": distortions,
                "sdop": sdop,
                "lower": lower,
                "upper": upper,
            }
        )

    figure, axis = plt.subplots(
        figsize=(9.3, 5.8)
    )

    marker_spacing = max(
        1,
        args.threshold_points // 10,
    )

    for index, series in enumerate(
        plotted_series
    ):
        line = axis.plot(
            thresholds,
            series["sdop"],
            label=series["label"],
            linestyle=series["linestyle"],
            marker=series["marker"],
            linewidth=series["linewidth"],
            markersize=(
                6.5 if index < 2 else 5.5
            ),
            markevery=marker_spacing,
            drawstyle="steps-post",
        )[0]

        if series["confidence_band"]:
            axis.fill_between(
                thresholds,
                series["lower"],
                series["upper"],
                step="post",
                alpha=0.10,
                color=line.get_color(),
                linewidth=0,
            )

    axis.axvline(
        args.primary_threshold,
        linestyle=":",
        linewidth=1.8,
        color="0.25",
        label=(
            rf"Primary "
            rf"$\delta={args.primary_threshold:g}$"
        ),
    )

    factorized_all = select_distortions(
        rows,
        method="factorized",
    )

    hybrid_all = select_distortions(
        rows,
        method="hybrid",
    )

    factorized_primary = sdop_at_threshold(
        factorized_all,
        args.primary_threshold,
    )

    hybrid_primary = sdop_at_threshold(
        hybrid_all,
        args.primary_threshold,
    )

    relative_reduction = (
        1.0
        - hybrid_primary
        / factorized_primary
        if factorized_primary > 0.0
        else math.nan
    )

    annotation = (
        rf"At $\delta={args.primary_threshold:g}$:"
        "\n"
        rf"Fixed = {factorized_primary:.3f}, "
        rf"AIF = {hybrid_primary:.3f}"
    )

    if math.isfinite(relative_reduction):
        annotation += (
            "\n"
            f"Relative reduction = "
            f"{100.0 * relative_reduction:.1f}%"
        )

    axis.text(
        0.985,
        0.035,
        annotation,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.88,
        },
    )

    axis.set_xlabel(
        r"Semantic distortion threshold $\delta$",
        fontsize=13,
    )

    axis.set_ylabel(
        "Semantic Distortion Outage Probability",
        fontsize=13,
    )

    axis.set_xlim(
        args.threshold_min,
        args.threshold_max,
    )

    axis.set_ylim(
        -0.02,
        1.02,
    )

    axis.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.75,
    )

    axis.legend(
        frameon=False,
        fontsize=9.5,
        loc="upper right",
    )

    axis.tick_params(
        axis="both",
        labelsize=11,
    )

    figure.tight_layout()

    output_stem = (
        args.out_dir
        / "sdop_curve_richer_by_action"
    )

    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    figure.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(figure)

    write_curve_csv(
        args.out_dir
        / "sdop_curve_richer_by_action.csv",
        plotted_series,
        thresholds,
    )

    print("\nCurves included")
    print("----------------")

    for series in plotted_series:
        primary_value = sdop_at_threshold(
            series["distortions"],
            args.primary_threshold,
        )

        print(
            f"{series['label']}: "
            f"SDOP({args.primary_threshold:g})="
            f"{primary_value:.4f}"
        )

    print("\nSaved files")
    print("-----------")

    for extension in (
        "png",
        "pdf",
        "svg",
    ):
        path = output_stem.with_suffix(
            f".{extension}"
        )

        print(path.resolve())

    print(
        (
            args.out_dir
            / "sdop_curve_richer_by_action.csv"
        ).resolve()
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare Fixed Factorized and Hybrid AIF using TOPIQ-FR.

This version creates a clean paired-event figure containing only:
- event-level observations;
- connecting lines between paired events.

The mean values, bootstrap confidence intervals, relative improvement,
and paired-test results are saved in CSV/TXT files but are not drawn
inside the figure.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

METHODS = ("factorized", "hybrid")
LABELS = {
    "factorized": "Fixed Factorized",
    "hybrid": "Hybrid AIF",
}

NAME_RE = re.compile(
    r"^(?P<object>.+)_(?P<method>reference|factorized|hybrid)_"
    r"step_(?P<step>\d+)_view_(?P<view>\d+)\.(?:png|jpg|jpeg|webp)$",
    re.IGNORECASE,
)

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Fixed Factorized and Hybrid AIF using TOPIQ-FR."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs_compare_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50"
        ),
        help="Directory containing reference, factorized, and hybrid images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs_topiq_clean"),
        help="Directory in which results will be saved.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    if args.permutation_samples < 1:
        parser.error("--permutation-samples must be at least 1")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    return args


def discover_images(run_dir: Path) -> Dict[Tuple[str, str, int, int], Path]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    index: Dict[Tuple[str, str, int, int], Path] = {}

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue

        match = NAME_RE.match(path.name)
        if match is None:
            continue

        key = (
            match.group("object"),
            match.group("method").lower(),
            int(match.group("step")),
            int(match.group("view")),
        )

        if key in index:
            raise RuntimeError(
                "Duplicate image key found:\n"
                f"  {index[key]}\n"
                f"  {path}"
            )

        index[key] = path

    if not any(key[1] == "reference" for key in index):
        raise RuntimeError(
            "No reference images were found. Expected names such as:\n"
            "<object>_reference_step_00_view_00.png"
        )

    return index


def make_pairs(
    index: Dict[Tuple[str, str, int, int], Path]
) -> List[Tuple[str, int, int, str, Path, Path]]:
    pairs: List[Tuple[str, int, int, str, Path, Path]] = []
    missing: List[str] = []

    reference_keys = sorted(
        key for key in index if key[1] == "reference"
    )

    for object_id, _, step, view in reference_keys:
        reference_path = index[(object_id, "reference", step, view)]

        for method in METHODS:
            candidate_path = index.get((object_id, method, step, view))

            if candidate_path is None:
                missing.append(
                    f"object={object_id}, method={method}, "
                    f"step={step}, view={view}"
                )
                continue

            pairs.append(
                (
                    object_id,
                    step,
                    view,
                    method,
                    reference_path,
                    candidate_path,
                )
            )

    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            "Some reference images are missing method outputs:\n"
            f"{preview}\n"
            f"Total missing pairs: {len(missing)}"
        )

    return pairs


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
    )


def batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_topiq(device: torch.device):
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError(
            "pyiqa is not installed. Run:\n"
            "python -m pip install -U pyiqa"
        ) from exc

    available_models = {
        model_name.lower()
        for model_name in pyiqa.list_models()
    }

    if "topiq_fr" not in available_models:
        raise RuntimeError(
            "The installed pyiqa version does not provide topiq_fr. "
            "Upgrade it with:\n"
            "python -m pip install -U pyiqa"
        )

    metric = pyiqa.create_metric(
        "topiq_fr",
        device=device,
    )
    metric.eval()

    return metric, bool(metric.lower_better)


def score_all_pairs(
    pairs: Sequence[Tuple[str, int, int, str, Path, Path]],
    metric,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for batch_number, pair_batch in enumerate(
        batched(pairs, batch_size),
        start=1,
    ):
        candidate_tensors: List[torch.Tensor] = []
        reference_tensors: List[torch.Tensor] = []
        expected_shape: Tuple[int, ...] | None = None

        for pair in pair_batch:
            (
                object_id,
                step,
                view,
                method,
                reference_path,
                candidate_path,
            ) = pair

            reference = load_rgb(reference_path)
            candidate = load_rgb(candidate_path)

            if candidate.shape != reference.shape:
                raise RuntimeError(
                    "Candidate and reference dimensions differ:\n"
                    f"  candidate: {candidate_path} {tuple(candidate.shape)}\n"
                    f"  reference: {reference_path} {tuple(reference.shape)}\n"
                    "TOPIQ-FR requires aligned image dimensions."
                )

            if expected_shape is None:
                expected_shape = tuple(candidate.shape)
            elif tuple(candidate.shape) != expected_shape:
                raise RuntimeError(
                    "Images in the same batch have different dimensions. "
                    "Rerun with --batch-size 1."
                )

            candidate_tensors.append(candidate)
            reference_tensors.append(reference)

        candidate_batch = torch.stack(candidate_tensors).to(device)
        reference_batch = torch.stack(reference_tensors).to(device)

        with torch.inference_mode():
            scores = metric(
                candidate_batch,
                reference_batch,
            )

        score_values = (
            scores.detach()
            .float()
            .reshape(-1)
            .cpu()
            .numpy()
        )

        if score_values.size != len(pair_batch):
            raise RuntimeError(
                "TOPIQ-FR returned an unexpected number of scores: "
                f"{score_values.size} for batch size {len(pair_batch)}."
            )

        for pair, score in zip(pair_batch, score_values):
            (
                object_id,
                step,
                view,
                method,
                reference_path,
                candidate_path,
            ) = pair

            rows.append(
                {
                    "object_id": object_id,
                    "step": step,
                    "view": view,
                    "method": method,
                    "reference_path": str(reference_path),
                    "candidate_path": str(candidate_path),
                    "topiq_fr": float(score),
                }
            )

        completed = min(
            batch_number * batch_size,
            len(pairs),
        )
        print(
            f"Scored {completed}/{len(pairs)} image pairs",
            flush=True,
        )

    return rows


def aggregate_events(
    image_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)

    for row in image_rows:
        key = (
            str(row["object_id"]),
            int(row["step"]),
            str(row["method"]),
        )
        grouped[key].append(float(row["topiq_fr"]))

    event_rows: List[Dict[str, object]] = []

    for (object_id, step, method), values in sorted(grouped.items()):
        event_rows.append(
            {
                "object_id": object_id,
                "step": step,
                "method": method,
                "num_views": len(values),
                "topiq_fr": float(np.mean(values)),
            }
        )

    return event_rows


def paired_arrays(
    event_rows: Sequence[Dict[str, object]],
) -> Tuple[List[Tuple[str, int]], np.ndarray, np.ndarray]:
    lookup = {
        (
            str(row["object_id"]),
            int(row["step"]),
            str(row["method"]),
        ): float(row["topiq_fr"])
        for row in event_rows
    }

    fixed_keys = {
        (object_id, step)
        for object_id, step, method in lookup
        if method == "factorized"
    }

    aif_keys = {
        (object_id, step)
        for object_id, step, method in lookup
        if method == "hybrid"
    }

    if fixed_keys != aif_keys:
        raise RuntimeError(
            "The event sets differ between Fixed Factorized and Hybrid AIF."
        )

    keys = sorted(fixed_keys)

    fixed = np.asarray(
        [
            lookup[(object_id, step, "factorized")]
            for object_id, step in keys
        ],
        dtype=np.float64,
    )

    aif = np.asarray(
        [
            lookup[(object_id, step, "hybrid")]
            for object_id, step in keys
        ],
        dtype=np.float64,
    )

    return keys, fixed, aif


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sample_count = len(values)

    bootstrap_means = rng.choice(
        values,
        size=(samples, sample_count),
        replace=True,
    ).mean(axis=1)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return (
        float(np.mean(values)),
        float(lower),
        float(upper),
    )


def permutation_pvalue(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(differences)))

    extreme_count = 0
    completed = 0
    chunk_size = 10000

    while completed < samples:
        current_size = min(
            chunk_size,
            samples - completed,
        )

        signs = rng.choice(
            (-1.0, 1.0),
            size=(current_size, len(differences)),
        )

        simulated = np.abs(
            (signs * differences).mean(axis=1)
        )

        extreme_count += int(
            np.sum(simulated >= observed - 1e-15)
        )
        completed += current_size

    return (
        extreme_count + 1.0
    ) / (
        samples + 1.0
    )


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def create_clean_figure(
    out_dir: Path,
    fixed: np.ndarray,
    aif: np.ndarray,
    lower_better: bool,
    dpi: int,
) -> None:
    """Create a clean paired-event figure without means or annotations."""

    figure, axis = plt.subplots(
        figsize=(3.5, 3.2)
    )

    fixed_x = 0.0
    aif_x = 1.0

    # Connect matching interaction events.
    for fixed_value, aif_value in zip(fixed, aif):
        axis.plot(
            [fixed_x, aif_x],
            [fixed_value, aif_value],
            linewidth=0.75,
            alpha=0.28,
            zorder=1,
        )

    # Plot the event-level observations only.
    axis.scatter(
        np.full(fixed.shape, fixed_x),
        fixed,
        s=20,
        alpha=0.60,
        edgecolors="none",
        zorder=2,
    )

    axis.scatter(
        np.full(aif.shape, aif_x),
        aif,
        s=20,
        alpha=0.60,
        edgecolors="none",
        zorder=2,
    )

    direction_text = (
        "lower is better"
        if lower_better
        else "higher is better"
    )

    axis.set_xticks(
        [fixed_x, aif_x],
        [LABELS["factorized"], LABELS["hybrid"]],
    )

    axis.set_ylabel(
        f"Event-level TOPIQ-FR\n({direction_text})",
        fontsize=8.5,
    )

    axis.set_xlim(-0.16, 1.16)

    all_values = np.concatenate([fixed, aif])
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    value_range = value_max - value_min
    padding = max(0.03, 0.07 * value_range)

    axis.set_ylim(
        value_min - padding,
        value_max + padding,
    )

    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.65,
    )

    axis.tick_params(
        axis="both",
        labelsize=7.5,
    )

    figure.tight_layout(pad=0.7)

    output_stem = (
        out_dir
        / "topiq_fr_factorized_vs_hybrid_aif_clean"
    )

    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda":
            print("CUDA is unavailable; using CPU instead.")
        device = torch.device("cpu")

    image_index = discover_images(args.run_dir)
    image_pairs = make_pairs(image_index)

    unique_images = {
        pair[4]
        for pair in image_pairs
    } | {
        pair[5]
        for pair in image_pairs
    }

    print(f"Paired image comparisons: {len(image_pairs)}")
    print(f"Unique images: {len(unique_images)}")
    print(f"Device: {device}")

    print("\nLoading TOPIQ-FR...")
    metric, lower_better = load_topiq(device)

    print(
        "Metric direction: "
        + (
            "lower is better"
            if lower_better
            else "higher is better"
        )
    )

    print("Calculating TOPIQ-FR...")
    image_rows = score_all_pairs(
        pairs=image_pairs,
        metric=metric,
        device=device,
        batch_size=args.batch_size,
    )

    event_rows = aggregate_events(image_rows)
    event_keys, fixed, aif = paired_arrays(event_rows)

    fixed_summary = bootstrap_mean_ci(
        fixed,
        args.bootstrap_samples,
        args.seed,
    )

    aif_summary = bootstrap_mean_ci(
        aif,
        args.bootstrap_samples,
        args.seed + 1,
    )

    raw_difference = aif - fixed
    oriented_improvement = (
        fixed - aif
        if lower_better
        else aif - fixed
    )

    improvement_summary = bootstrap_mean_ci(
        oriented_improvement,
        args.bootstrap_samples,
        args.seed + 2,
    )

    p_value = permutation_pvalue(
        raw_difference,
        args.permutation_samples,
        args.seed + 3,
    )

    fixed_mean = fixed_summary[0]
    aif_mean = aif_summary[0]

    if lower_better:
        relative_improvement = (
            100.0
            * (fixed_mean - aif_mean)
            / abs(fixed_mean)
        )
    else:
        relative_improvement = (
            100.0
            * (aif_mean - fixed_mean)
            / abs(fixed_mean)
        )

    paired_rows: List[Dict[str, object]] = []

    for (
        (object_id, step),
        fixed_value,
        aif_value,
        oriented_gain,
    ) in zip(
        event_keys,
        fixed,
        aif,
        oriented_improvement,
    ):
        paired_rows.append(
            {
                "object_id": object_id,
                "step": step,
                "fixed_factorized": float(fixed_value),
                "hybrid_aif": float(aif_value),
                "aif_minus_fixed": float(
                    aif_value - fixed_value
                ),
                "aif_oriented_improvement": float(
                    oriented_gain
                ),
            }
        )

    write_csv(
        args.out_dir / "topiq_fr_per_image.csv",
        image_rows,
    )

    write_csv(
        args.out_dir / "topiq_fr_per_event.csv",
        event_rows,
    )

    write_csv(
        args.out_dir / "topiq_fr_paired_events.csv",
        paired_rows,
    )

    summary_rows = [
        {
            "num_paired_events": len(event_keys),
            "lower_better": lower_better,
            "fixed_mean": fixed_summary[0],
            "fixed_ci_low": fixed_summary[1],
            "fixed_ci_high": fixed_summary[2],
            "aif_mean": aif_summary[0],
            "aif_ci_low": aif_summary[1],
            "aif_ci_high": aif_summary[2],
            "mean_aif_minus_fixed": float(
                np.mean(raw_difference)
            ),
            "oriented_improvement": improvement_summary[0],
            "improvement_ci_low": improvement_summary[1],
            "improvement_ci_high": improvement_summary[2],
            "relative_improvement_percent": relative_improvement,
            "paired_randomization_pvalue": p_value,
            "aif_wins": int(
                np.sum(oriented_improvement > 0)
            ),
            "ties": int(
                np.sum(
                    np.isclose(oriented_improvement, 0.0)
                )
            ),
            "aif_losses": int(
                np.sum(oriented_improvement < 0)
            ),
        }
    ]

    write_csv(
        args.out_dir / "topiq_fr_summary.csv",
        summary_rows,
    )

    summary_text = (
        "TOPIQ-FR comparison\n"
        "===================\n"
        f"Paired events: {len(event_keys)}\n"
        "Direction: "
        f"{'lower is better' if lower_better else 'higher is better'}\n"
        f"Fixed Factorized: {fixed_summary[0]:.6f} "
        f"[95% CI {fixed_summary[1]:.6f}, "
        f"{fixed_summary[2]:.6f}]\n"
        f"Hybrid AIF:       {aif_summary[0]:.6f} "
        f"[95% CI {aif_summary[1]:.6f}, "
        f"{aif_summary[2]:.6f}]\n"
        f"AIF relative improvement: {relative_improvement:.2f}%\n"
        "Paired improvement 95% CI: "
        f"[{improvement_summary[1]:.6f}, "
        f"{improvement_summary[2]:.6f}]\n"
        f"Paired randomization p-value: {p_value:.8g}\n"
        "AIF wins/ties/losses: "
        f"{int(np.sum(oriented_improvement > 0))}/"
        f"{int(np.sum(np.isclose(oriented_improvement, 0.0)))}/"
        f"{int(np.sum(oriented_improvement < 0))}\n"
    )

    summary_path = (
        args.out_dir
        / "topiq_fr_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    create_clean_figure(
        out_dir=args.out_dir,
        fixed=fixed,
        aif=aif,
        lower_better=lower_better,
        dpi=args.dpi,
    )

    print("\n" + summary_text)
    print("Saved files")
    print("-----------")

    for filename in (
        "topiq_fr_factorized_vs_hybrid_aif_clean.png",
        "topiq_fr_factorized_vs_hybrid_aif_clean.pdf",
        "topiq_fr_factorized_vs_hybrid_aif_clean.svg",
        "topiq_fr_per_image.csv",
        "topiq_fr_per_event.csv",
        "topiq_fr_paired_events.csv",
        "topiq_fr_summary.csv",
        "topiq_fr_summary.txt",
    ):
        print((args.out_dir / filename).resolve())


if __name__ == "__main__":
    main()

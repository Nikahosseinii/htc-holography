import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, shift


def load_phase_bitmap(path: str, max_size: int = 1024) -> np.ndarray:
    """
    Load an 8-bit SLM phase bitmap and convert it to radians in [0, 2pi).

    The optional resize keeps the FFT memory small for quick testing.
    """
    img = Image.open(path).convert("L")

    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    phase = arr * 2.0 * np.pi
    return phase


def normalize_intensity(I: np.ndarray) -> np.ndarray:
    I = I.astype(np.float32)
    return I / (I.max() + 1e-12)


def propagate_far_field(phase: np.ndarray, amplitude: np.ndarray | None = None) -> np.ndarray:
    """
    Simple Fourier-plane replay model.

    This is a software-side proxy for SLM replay. It is enough for a
    pseudo-hardware sensitivity test.
    """
    if amplitude is None:
        amplitude = np.ones_like(phase, dtype=np.float32)

    field = amplitude * np.exp(1j * phase)
    replay = np.fft.fftshift(np.fft.fft2(field))
    intensity = np.abs(replay) ** 2
    return normalize_intensity(intensity)


def gaussian_beam_amplitude(shape, sigma_frac: float = 0.35) -> np.ndarray:
    """
    Nonuniform incident illumination model.
    """
    h, w = shape
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    r2 = xx**2 + yy**2
    amp = np.exp(-r2 / (2.0 * sigma_frac**2))
    return (amp / amp.max()).astype(np.float32)


def wavefront_aberration(shape, strength_rad: float = 0.75) -> np.ndarray:
    """
    Simple low-order wavefront aberration model.
    """
    h, w = shape
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]

    defocus = 2.0 * (xx**2 + yy**2) - 1.0
    astigmatism = xx**2 - yy**2
    coma = (3.0 * (xx**2 + yy**2) - 2.0) * xx

    aberr = 0.50 * defocus + 0.30 * astigmatism + 0.20 * coma
    aberr = aberr / (np.max(np.abs(aberr)) + 1e-12)
    return (strength_rad * aberr).astype(np.float32)


def simulate_replay_case(phase: np.ndarray, case: str) -> np.ndarray:
    """
    Simulate one hardware-facing perturbation.
    """
    amplitude = np.ones_like(phase, dtype=np.float32)
    phase_eff = phase.copy()

    if case == "ideal":
        return propagate_far_field(phase_eff, amplitude)

    if case in {"nonuniform_illumination", "all_uncalibrated"}:
        amplitude = gaussian_beam_amplitude(phase.shape, sigma_frac=0.35)

    if case in {"wavefront_error", "all_uncalibrated"}:
        phase_eff = phase_eff + wavefront_aberration(phase.shape, strength_rad=0.75)

    if case in {"pixel_crosstalk", "all_uncalibrated"}:
        # Proxy for LCOS neighboring-pixel crosstalk:
        # the displayed phase becomes locally smoothed.
        phase_eff = gaussian_filter(phase_eff, sigma=0.75)

    intensity = propagate_far_field(phase_eff, amplitude)

    if case in {"misalignment", "all_uncalibrated"}:
        intensity = shift(intensity, shift=(6, -4), mode="nearest")
        intensity = normalize_intensity(intensity)

    return intensity


def nrmse(test: np.ndarray, ref: np.ndarray) -> float:
    denom = ref.max() - ref.min() + 1e-12
    return float(np.sqrt(np.mean((test - ref) ** 2)) / denom)


def pearson_corr(test: np.ndarray, ref: np.ndarray) -> float:
    a = test.reshape(-1)
    b = ref.reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])


def save_intensity(path: Path, I: np.ndarray) -> None:
    img = (255.0 * normalize_intensity(I)).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def make_montage(out_dir: Path, cases: list[str]) -> None:
    images = []
    labels = []

    for case in cases:
        p = out_dir / f"{case}_reconstruction.png"
        img = Image.open(p).convert("RGB").resize((280, 280))
        images.append(img)
        labels.append(case.replace("_", " "))

    pad = 20
    label_h = 35
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    W = cols * 280 + (cols + 1) * pad
    H = rows * (280 + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = pad + c * (280 + pad)
        y = pad + r * (280 + label_h + pad)
        canvas.paste(img, (x, y + label_h))
        draw.text((x, y + 8), labels[i], fill=(0, 0, 0))

    canvas.save(out_dir / "slm_stress_reconstruction_montage.png")


def plot_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    df_plot = df[df["case"] != "ideal"].copy()

    plt.figure(figsize=(8.8, 5.0))
    plt.bar(df_plot["case"], df_plot["nrmse_vs_ideal"])
    plt.ylabel("NRMSE vs. ideal replay")
    plt.xlabel("Simulated hardware effect")
    plt.title("Pseudo-hardware sensitivity of SLM phase map")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "slm_stress_nrmse.png", dpi=300)
    plt.savefig(out_dir / "slm_stress_nrmse.pdf")
    plt.savefig(out_dir / "slm_stress_nrmse.svg")
    plt.close()

    plt.figure(figsize=(8.8, 5.0))
    plt.bar(df_plot["case"], df_plot["correlation_vs_ideal"])
    plt.ylabel("Correlation vs. ideal replay")
    plt.xlabel("Simulated hardware effect")
    plt.title("Pseudo-hardware replay correlation")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "slm_stress_correlation.png", dpi=300)
    plt.savefig(out_dir / "slm_stress_correlation.pdf")
    plt.savefig(out_dir / "slm_stress_correlation.svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="Path to SLM phase bitmap, e.g., slm_phase_3840x2160.png")
    parser.add_argument("--out_dir", default="slm_pseudo_hardware_stress_test")
    parser.add_argument("--max_size", type=int, default=1024, help="Resize largest dimension for fast FFT")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase = load_phase_bitmap(args.phase, max_size=args.max_size)

    cases = [
        "ideal",
        "nonuniform_illumination",
        "wavefront_error",
        "pixel_crosstalk",
        "misalignment",
        "all_uncalibrated",
    ]

    ideal = simulate_replay_case(phase, "ideal")
    save_intensity(out_dir / "ideal_reconstruction.png", ideal)

    rows = []
    for case in cases:
        I = simulate_replay_case(phase, case)
        save_intensity(out_dir / f"{case}_reconstruction.png", I)

        rows.append({
            "case": case,
            "nrmse_vs_ideal": nrmse(I, ideal),
            "correlation_vs_ideal": pearson_corr(I, ideal),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "slm_stress_metrics.csv", index=False)

    plot_metrics(df, out_dir)
    make_montage(out_dir, cases)

    print("\n=== SLM pseudo-hardware stress test ===")
    print(df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

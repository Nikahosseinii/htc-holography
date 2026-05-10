#!/usr/bin/env python3
"""
mesh_to_slm_gaea_phase.py

Convert a reconstructed 3D mesh into an SLM-format phase-only CGH image
for the same SLM used in the Chlipala et al. Scientific Reports 2025 paper:

Target SLM:
- HoloEye Gaea 2.0
- phase-only LCoS SLM
- resolution: 3840 x 2160
- pixel pitch: 3.74 um

This script produces:
- slm_phase_3840x2160.png  : 8-bit grayscale phase image
- slm_phase_3840x2160.bmp  : same phase image in BMP format
- phase_radians.npy        : phase in radians, range [-pi, pi]
- phase_0_2pi.npy          : phase in radians, range [0, 2pi)
- reconstructed_intensity.png
- depth_slice_preview.png
- summary.png
- metadata.json

Important:
This creates an SLM-format phase map, but a real hardware experiment
still requires device-specific phase calibration/LUT and optical alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


# ============================================================
# Mesh loading and sampling
# ============================================================

def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"No geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh type: {type(loaded)}")

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Mesh is empty: {mesh_path}")

    return mesh


def normalize_mesh_to_unit_box(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalize mesh to fit inside roughly [-0.5, 0.5]^3.
    """
    mesh = mesh.copy()
    verts = mesh.vertices.astype(np.float32)

    center = verts.mean(axis=0, keepdims=True)
    verts = verts - center

    max_extent = np.max(np.ptp(verts, axis=0))
    if max_extent <= 0:
        raise ValueError("Mesh scale is zero.")

    verts = verts / max_extent
    mesh.vertices = verts

    return mesh


def sample_surface_points_and_colors(mesh: trimesh.Trimesh, n_points: int):
    """
    Returns:
        points: (N, 3), normalized xyz
        colors: (N, 3), RGB in [0,1]

    If the mesh has no usable colors, uses white amplitudes.
    """
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)

    colors = np.ones((n_points, 3), dtype=np.float32)

    try:
        # Case 1: face colors
        if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
            fc = np.asarray(mesh.visual.face_colors)
            if fc.ndim == 2 and fc.shape[0] >= mesh.faces.shape[0]:
                c = fc[face_idx, :3].astype(np.float32) / 255.0
                colors = np.clip(c, 0.0, 1.0)

        # Case 2: vertex colors
        elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] >= mesh.vertices.shape[0]:
                face_vertices = mesh.faces[face_idx]
                c = vc[face_vertices, :3].astype(np.float32).mean(axis=1) / 255.0
                colors = np.clip(c, 0.0, 1.0)

    except Exception:
        colors = np.ones((n_points, 3), dtype=np.float32)

    return points, colors


# ============================================================
# Convert sampled mesh into depth slices
# ============================================================

def gaussian_blur_stack(stack: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return stack

    import torch.nn.functional as F

    device = stack.device
    dtype = stack.dtype

    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    sigma = max(kernel_size / 5.0, 1.0)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kx = g.view(1, 1, 1, kernel_size)
    ky = g.view(1, 1, kernel_size, 1)

    y = stack.unsqueeze(1)
    y = F.conv2d(y, kx, padding=(0, kernel_size // 2))
    y = F.conv2d(y, ky, padding=(kernel_size // 2, 0))

    return y.squeeze(1)


def points_to_depth_slices(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    height: int,
    width: int,
    n_slices: int,
    fill_fraction: float,
    color_channel: str,
    device: torch.device,
):
    """
    Rasterize sampled 3D points into amplitude layers.

    The mesh is normalized to fit the SLM aspect ratio while preserving shape.
    """
    pts = torch.from_numpy(points_xyz).to(device=device, dtype=torch.float32)
    cols = torch.from_numpy(colors_rgb).to(device=device, dtype=torch.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Center again just in case.
    x = x - x.mean()
    y = y - y.mean()
    z = z - z.mean()

    # Fit x/y into SLM canvas.
    # Mesh coordinates are normalized, then scaled by fill_fraction.
    aspect = width / height

    x_range = (x.max() - x.min()).clamp(min=1e-8)
    y_range = (y.max() - y.min()).clamp(min=1e-8)

    # Determine scale so object fits in both dimensions.
    scale_x = fill_fraction / x_range
    scale_y = fill_fraction / y_range
    scale = torch.minimum(scale_x, scale_y)

    x_norm = x * scale
    y_norm = y * scale

    # Convert to [0,1].
    x01 = torch.clamp(x_norm + 0.5, 0.0, 1.0)
    y01 = torch.clamp(0.5 - y_norm, 0.0, 1.0)

    # Depth ordering to [0,1].
    z01 = (z - z.min()) / (z.max() - z.min()).clamp(min=1e-8)
    z01 = torch.clamp(z01, 0.0, 1.0)

    px = torch.clamp((x01 * (width - 1)).long(), 0, width - 1)
    py = torch.clamp((y01 * (height - 1)).long(), 0, height - 1)
    sid = torch.clamp((z01 * (n_slices - 1)).long(), 0, n_slices - 1)

    if color_channel == "red":
        values = cols[:, 0]
    elif color_channel == "green":
        values = cols[:, 1]
    elif color_channel == "blue":
        values = cols[:, 2]
    elif color_channel == "luma":
        values = 0.2126 * cols[:, 0] + 0.7152 * cols[:, 1] + 0.0722 * cols[:, 2]
    else:
        values = torch.ones_like(x)

    values = torch.clamp(values, 0.0, 1.0)

    amplitudes = torch.zeros(
        (n_slices, height, width),
        dtype=torch.float32,
        device=device,
    )

    flat = amplitudes.view(-1)
    flat_idx = sid * (height * width) + py * width + px
    flat.scatter_add_(0, flat_idx, values)

    amplitudes = flat.view(n_slices, height, width)

    max_val = amplitudes.max().clamp(min=1e-8)
    amplitudes = amplitudes / max_val

    amplitudes = gaussian_blur_stack(amplitudes, kernel_size=5)
    amplitudes = amplitudes / amplitudes.max().clamp(min=1e-8)

    return amplitudes


# ============================================================
# Fresnel propagation and CGH
# ============================================================

def fresnel_propagate(
    field: torch.Tensor,
    wavelength_m: float,
    pixel_pitch_m: float,
    z_m: float,
) -> torch.Tensor:
    """
    Fresnel propagation using transfer function approximation.
    """
    if z_m <= 0:
        return field

    h, w = field.shape
    device = field.device

    fx = torch.fft.fftfreq(w, d=pixel_pitch_m, device=device)
    fy = torch.fft.fftfreq(h, d=pixel_pitch_m, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    phase = -math.pi * wavelength_m * z_m * (FX ** 2 + FY ** 2)
    H = torch.exp(1j * phase).to(torch.complex64)

    U1 = torch.fft.fft2(field)
    U2 = U1 * H
    out = torch.fft.ifft2(U2)

    return out


def build_phase_hologram(
    amplitudes: torch.Tensor,
    wavelength_m: float,
    pixel_pitch_m: float,
    z_near_m: float,
    z_far_m: float,
    random_phase: bool,
):
    """
    Sum propagated fields from depth slices, then extract phase-only hologram.
    """
    n_slices, h, w = amplitudes.shape
    device = amplitudes.device

    total = torch.zeros((h, w), dtype=torch.complex64, device=device)
    z_positions = torch.linspace(z_near_m, z_far_m, n_slices, device=device)

    for i in range(n_slices):
        amp = amplitudes[i]

        if random_phase:
            phase0 = 2.0 * math.pi * torch.rand_like(amp)
            obj_field = amp.to(torch.complex64) * torch.exp(1j * phase0).to(torch.complex64)
        else:
            obj_field = amp.to(torch.complex64)

        propagated = fresnel_propagate(
            field=obj_field,
            wavelength_m=wavelength_m,
            pixel_pitch_m=pixel_pitch_m,
            z_m=float(z_positions[i].item()),
        )

        total = total + propagated

    phase = torch.angle(total)
    phase_0_2pi = torch.remainder(phase + 2.0 * math.pi, 2.0 * math.pi)

    return total, phase, phase_0_2pi, z_positions


def phase_to_uint8(phase_0_2pi: np.ndarray, lut_path: str | None = None) -> np.ndarray:
    """
    Convert phase in [0, 2pi) to 8-bit grayscale.

    If a LUT is provided, it should be a .npy file with shape (256,)
    mapping linear phase gray values to calibrated SLM gray values.
    """
    gray = np.floor(255.0 * phase_0_2pi / (2.0 * np.pi)).astype(np.uint8)

    if lut_path:
        lut = np.load(lut_path).astype(np.uint8)
        if lut.shape[0] != 256:
            raise ValueError("LUT must have shape (256,)")
        gray = lut[gray]

    return gray


# ============================================================
# Save outputs
# ============================================================

def save_outputs(
    out_dir: str,
    amplitudes: torch.Tensor,
    field: torch.Tensor,
    phase: torch.Tensor,
    phase_0_2pi: torch.Tensor,
    phase_u8: np.ndarray,
    metadata: dict,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    phase_np = phase.detach().cpu().numpy()
    phase_0_2pi_np = phase_0_2pi.detach().cpu().numpy()
    field_np = field.detach().cpu().numpy()

    np.save(out / "phase_radians.npy", phase_np)
    np.save(out / "phase_0_2pi.npy", phase_0_2pi_np)

    imageio.imwrite(out / "slm_phase_3840x2160.png", phase_u8)
    imageio.imwrite(out / "slm_phase_3840x2160.bmp", phase_u8)

    intensity = np.abs(field_np) ** 2
    intensity = intensity / (intensity.max() + 1e-8)
    intensity_u8 = (255.0 * intensity).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out / "reconstructed_intensity.png", intensity_u8)

    amp_np = amplitudes.detach().cpu().numpy()
    middle = amp_np[amp_np.shape[0] // 2]
    middle_u8 = (255.0 * middle / (middle.max() + 1e-8)).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out / "depth_slice_preview.png", middle_u8)

    # Save a few low-res slice previews.
    preview_dir = out / "slice_previews"
    preview_dir.mkdir(exist_ok=True)

    n_save = min(8, amp_np.shape[0])
    idxs = np.linspace(0, amp_np.shape[0] - 1, n_save).round().astype(int)

    for k, idx in enumerate(idxs):
        img = amp_np[idx]
        img_u8 = (255.0 * img / (img.max() + 1e-8)).clip(0, 255).astype(np.uint8)
        imageio.imwrite(preview_dir / f"slice_{idx:02d}.png", img_u8)

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Summary figure.
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(middle_u8, cmap="gray")
    plt.title("Middle depth slice")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(intensity_u8, cmap="gray")
    plt.title("Reconstructed intensity")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(phase_u8, cmap="gray")
    plt.title("SLM phase map, 8-bit")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a mesh into a HoloEye Gaea 2.0 SLM-format phase-only CGH."
    )

    parser.add_argument("--mesh", type=str, required=True, help="Input mesh path, e.g., .obj/.ply/.glb/.stl")
    parser.add_argument("--out", type=str, required=True, help="Output directory")

    parser.add_argument("--width", type=int, default=3840, help="SLM width in pixels")
    parser.add_argument("--height", type=int, default=2160, help="SLM height in pixels")
    parser.add_argument("--pixel_pitch_um", type=float, default=3.74, help="SLM pixel pitch in micrometers")

    parser.add_argument(
        "--wavelength_nm",
        type=float,
        default=515.0,
        help="Wavelength in nm. For the paper's green LED use 515 nm.",
    )

    parser.add_argument("--points", type=int, default=250000, help="Number of mesh surface points to sample")
    parser.add_argument("--slices", type=int, default=16, help="Number of depth slices")

    parser.add_argument("--z_near_mm", type=float, default=50.0, help="Nearest propagation distance in mm")
    parser.add_argument("--z_far_mm", type=float, default=150.0, help="Farthest propagation distance in mm")

    parser.add_argument(
        "--fill_fraction",
        type=float,
        default=0.70,
        help="Fraction of SLM area occupied by the object amplitude support",
    )

    parser.add_argument(
        "--color_channel",
        choices=["red", "green", "blue", "luma", "white"],
        default="green",
        help="Amplitude channel used for the hologram. Use green for 515 nm proof-of-concept.",
    )

    parser.add_argument(
        "--random_phase",
        action="store_true",
        help="Add random phase to each depth slice before propagation. Can reduce artifacts.",
    )

    parser.add_argument(
        "--lut",
        type=str,
        default="",
        help="Optional .npy SLM calibration LUT with shape (256,). Leave empty for linear mapping.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")

    args = parser.parse_args()

    if args.width != 3840 or args.height != 2160:
        print("[WARN] The paper's HoloEye Gaea 2.0 SLM is 3840 x 2160. You changed the size.")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading mesh: {args.mesh}")

    mesh = load_mesh(args.mesh)
    mesh = normalize_mesh_to_unit_box(mesh)

    print(f"Sampling {args.points} mesh surface points...")
    points, colors = sample_surface_points_and_colors(mesh, args.points)

    print("Rasterizing sampled points into full-resolution depth slices...")
    amplitudes = points_to_depth_slices(
        points_xyz=points,
        colors_rgb=colors,
        height=args.height,
        width=args.width,
        n_slices=args.slices,
        fill_fraction=args.fill_fraction,
        color_channel=args.color_channel,
        device=device,
    )

    wavelength_m = args.wavelength_nm * 1e-9
    pixel_pitch_m = args.pixel_pitch_um * 1e-6
    z_near_m = args.z_near_mm * 1e-3
    z_far_m = args.z_far_mm * 1e-3

    print("Computing phase-only CGH...")
    field, phase, phase_0_2pi, z_positions = build_phase_hologram(
        amplitudes=amplitudes,
        wavelength_m=wavelength_m,
        pixel_pitch_m=pixel_pitch_m,
        z_near_m=z_near_m,
        z_far_m=z_far_m,
        random_phase=args.random_phase,
    )

    print("Converting phase to 8-bit SLM grayscale...")
    phase_u8 = phase_to_uint8(
        phase_0_2pi.detach().cpu().numpy(),
        lut_path=args.lut if args.lut else None,
    )

    metadata = {
        "input_mesh": args.mesh,
        "target_slm": "HoloEye Gaea 2.0 phase-only LCoS SLM",
        "width_px": args.width,
        "height_px": args.height,
        "pixel_pitch_um": args.pixel_pitch_um,
        "wavelength_nm": args.wavelength_nm,
        "color_channel": args.color_channel,
        "n_points": args.points,
        "n_slices": args.slices,
        "z_near_mm": args.z_near_mm,
        "z_far_mm": args.z_far_mm,
        "fill_fraction": args.fill_fraction,
        "random_phase": bool(args.random_phase),
        "lut_used": args.lut if args.lut else "linear 0..255 mapping, not hardware calibrated",
        "outputs": {
            "slm_phase_png": "slm_phase_3840x2160.png",
            "slm_phase_bmp": "slm_phase_3840x2160.bmp",
            "phase_radians": "phase_radians.npy",
            "phase_0_2pi": "phase_0_2pi.npy",
            "reconstructed_intensity": "reconstructed_intensity.png",
            "summary": "summary.png",
        },
        "note": (
            "This is an SLM-format phase-only CGH. Real optical replay still requires "
            "the device-specific phase calibration LUT, RGB illumination alignment, "
            "and the optical system calibration."
        ),
    }

    print("Saving outputs...")
    save_outputs(
        out_dir=args.out,
        amplitudes=amplitudes,
        field=field,
        phase=phase,
        phase_0_2pi=phase_0_2pi,
        phase_u8=phase_u8,
        metadata=metadata,
    )

    print(f"Done. Files saved in: {args.out}")
    print("Main SLM-ready files:")
    print(f"  {args.out}/slm_phase_3840x2160.png")
    print(f"  {args.out}/slm_phase_3840x2160.bmp")


if __name__ == "__main__":
    main()

# Figure 12: SLM-Ready Phase-Only Hologram

This folder contains the code used to recreate Figure 12 of the paper.

Figure 12 shows the software-side SLM hologram generation result, including:

- middle depth slice,
- reconstructed intensity,
- SLM phase map in 8-bit format.

The script converts a reconstructed 3D mesh into an SLM-ready phase-only hologram for a GAEA-like SLM configuration.

## Script

- `mesh_to_slm_gaea_phase.py`: generates the SLM-ready phase map and summary figure from an input mesh.

## Recreate Figure 12

From the MVDream working directory:
## Running the enhanced Fig. 12 script

To regenerate the enhanced Fig. 12 SLM-oriented hologram preparation results, run:

```bash
python mesh_to_slm_gaea_phase_enhanced_fig12.py \
  --mesh /mnt/d/object1_2mv.obj \
  --out ~/holo_output_gaea_green_enhanced_fig12 \
  --width 3840 \
  --height 2160 \
  --pixel_pitch_um 3.74 \
  --wavelength_nm 515 \
  --points 250000 \
  --slices 16 \
  --z_near_mm 50 \
  --z_far_mm 150 \
  --fill_fraction 0.70 \
  --color_channel green \
  --device cuda
The script saves outputs to:

/mnt/e/holo_output_gaea_green

Main files:

summary.png
depth_slice_preview.png
reconstructed_intensity.png
slm_phase_3840x2160.png
slm_phase_3840x2160.bmp

The main Figure 12 file is:

summary.png

The SLM-ready files are:

slm_phase_3840x2160.png
slm_phase_3840x2160.bmp


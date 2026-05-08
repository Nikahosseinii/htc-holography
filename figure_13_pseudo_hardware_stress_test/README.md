# Figure 13: Pseudo-Hardware Stress Test for SLM Replay

This folder contains the code and outputs used to generate Figure 13 of the paper.

The experiment starts from an SLM-ready phase-only hologram and evaluates a software pseudo-hardware replay model under common SLM-facing perturbations:

- nonuniform illumination
- wavefront phase error
- pixel-crosstalk-induced phase smoothing
- reconstruction-plane misalignment
- all perturbations jointly

The main output figure compares each perturbation against the ideal software replay using:

- NRMSE vs. ideal replay
- Pearson correlation vs. ideal replay

## Files

- `slm_pseudo_hardware_stress_test.py`: runs the pseudo-hardware stress test and creates the metrics CSV.
- `plot_slm_stress_one_figure.py`: creates the final Figure 13 line graph.
- `slm_stress_metrics.csv`: numerical results.
- `slm_stress_one_figure.png`: Figure 13 in PNG format.
- `slm_stress_one_figure.pdf`: Figure 13 in PDF format.
- `slm_stress_one_figure.svg`: Figure 13 in SVG format.
- `slm_stress_reconstruction_montage.png`: montage of replay reconstructions.

## Run

From the main project directory:

```bash
python slm_pseudo_hardware_stress_test.py \
  --phase /mnt/e/holo_output_gaea_green/slm_phase_3840x2160.png \
  --out_dir slm_pseudo_hardware_stress_test \
  --max_size 1024

python plot_slm_stress_one_figure.py \
  --csv slm_pseudo_hardware_stress_test/slm_stress_metrics.csv \
  --out_prefix slm_pseudo_hardware_stress_test/slm_stress_one_figure

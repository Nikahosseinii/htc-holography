# SLM-ready hologram output

This folder contains the software-side SLM-oriented hologram output generated from the reconstructed 3D mesh.

Files:

- `summary.png`: visualization showing the depth slice, reconstructed intensity, and 8-bit SLM phase map.
- `slm_phase_3840x2160.bmp`: grayscale phase-only bitmap intended as the SLM input pattern.
- `slm_phase_3840x2160.png`: PNG version of the same phase pattern for visualization.
- `metadata.json`: generation parameters, including SLM resolution, pixel pitch, wavelength, number of depth slices, and propagation range.

The target SLM geometry follows the HoloEye Gaea 2.0 phase-only LCoS SLM used in Chlipala et al., Scientific Reports 2025:
resolution 3840 x 2160 and pixel pitch 3.74 micrometers.

Note: optical replay still requires device-specific phase calibration/LUT and optical alignment.

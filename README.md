# HTC Holography

This repository contains reproducible code and supplementary materials for our paper on **JEPA-inspired factorized holographic-type communication (HTC)**.

The project studies how multi-view generative models can be adapted for interactive holographic communication, where users continuously change viewpoint or request local edits. Instead of repeatedly transmitting or regenerating full high-dimensional visual content, the proposed approach separates a scene into:

- a shared latent representation for stable object identity and geometry, and
- lightweight view-dependent or instruction-dependent updates for interaction.

The repository includes scripts for reproducing the main figures, metrics, and supplementary experiments reported in the manuscript.

---

## Repository Structure

### Core evaluation scripts

| Folder | Description |
|---|---|
| `CLIP/` | CLIP-score evaluation scripts for semantic alignment. |
| `IS/` | Inception Score evaluation scripts for generated multi-view outputs. |
| `IBR/` | Interactive Bandwidth Reduction payload calculations and related scripts. |
| `generation/` | Baseline and factorized generation scripts. |
| `scripts/` | Helper scripts for reproducing selected results. |

### Figure reproduction folders

| Folder | Figure | Description |
|---|---:|---|
| `Figure3_ILS_Plotting_Code/` | Fig. 3 | Instruction Locality Score plotting code. |
| `Figure4_Cumulative_Transmitted_Data/` | Fig. 4 | Cumulative transmitted data and bandwidth comparison. |
| `Figure5_Interaction_Latency/` | Fig. 5 | Interaction-latency plotting code. |
| `Figure6_Qualitative_Comparison/` | Fig. 6 | Qualitative comparison figure code. |
| `figure_7_hybrid_aif_qualitative/` | Fig. 7 | Hybrid AIF qualitative multi-view montage code. |
| `figure_8_hybrid_aif_ils/` | Fig. 8 | Hybrid AIF Instruction Locality Score code. |
| `figure_9_strongly_adaptive_regret/` | Fig. 9 | Strongly adaptive regret computation and plotting code. |
| `figure_10_paue_pafue/` | Fig. 10 | Prediction-aware update error and final update error code. |
| `figure_11_multiview_to_3d_mesh/` | Fig. 11 | MVDream multi-view outputs to Hunyuan3D-2mv mesh reconstruction and figure assembly. |
| `figure_12_slm_phase_hologram/` | Fig. 12 | SLM-ready phase-only hologram generation code. |
| `figure_13_pseudo_hardware_stress_test/` | Fig. 13 | Pseudo-hardware SLM stress-test code. |

### Supplementary materials

| Folder | Description |
|---|---|
| `supplementary/` | Supplementary meshes and hologram-related artifacts. |

---

## Main Experiments

### Static generation quality

The repository includes evaluation scripts for:

- CLIP score,
- Inception Score,
- camera-update stability,
- qualitative multi-view consistency.

These scripts are used to compare standard MVDream-style generation with the proposed factorized and Hybrid AIF variants.

### Interactive communication metrics

The repository includes code for evaluating:

- **IBR**: Interactive Bandwidth Reduction,
- **ILS**: Instruction Locality Score,
- **CUS**: Camera-Update Stability,
- **SA-Regret**: Strongly Adaptive Regret,
- **PAUE**: Prediction-Aware Update Error,
- **PAFUE**: Prediction-Aware Final Update Error.

These metrics evaluate semantic fidelity, interaction locality, communication efficiency, and policy adaptation over user-driven viewpoint changes.

### 2D-to-3D and hologram pipeline

The later figure folders reproduce the bridge from multi-view image generation to 3D and holographic display preparation:

1. MVDream generates four conditioning views.
2. Hunyuan3D-2mv reconstructs a 3D mesh from the multi-view inputs.
3. The mesh is rendered from multiple viewpoints.
4. A phase-only SLM hologram is generated.
5. Pseudo-hardware perturbations are applied to test robustness.

---

## Reproducing Individual Figures

Each figure folder contains its own `README.md` with the commands needed to reproduce that figure.

For example:

```bash
cd figure_10_paue_pafue
cat README.md

or:

cd figure_11_multiview_to_3d_mesh
cat README.md

Most scripts assume that the original working directories are available locally, such as:

~/MVDream
~/Hunyuan3D-2

Some paths in the scripts point to mounted Windows drives under WSL, such as:

/mnt/d/
/mnt/e/

Update these paths if your local setup is different.

Environment Notes

The experiments were run primarily under Ubuntu/WSL with NVIDIA GPU support.

Common environments used in the experiments include:

source ~/MVDream/.venv310/bin/activate
source ~/MVDream/.venv_ils/bin/activate
source ~/.venv_hy3d/bin/activate

The Hunyuan3D-2mv mesh-generation scripts require the Hunyuan3D environment and model weights. The first run may download model files from Hugging Face.

Outputs

Generated outputs are intentionally not all stored in this repository because many image, mesh, and hologram files can be large.

Typical outputs include:

.png, .pdf, and .svg figures,
.csv metric summaries,
.json adaptive-regret results,
.glb or .obj reconstructed meshes,
SLM phase maps such as .png and .bmp.

Each figure-specific README explains where the expected outputs are saved.

Citation

If you use this code or build on this project, please cite the corresponding paper once the final citation information is available.

Contact

For questions about the code or reproducibility, please open an issue in this repository.

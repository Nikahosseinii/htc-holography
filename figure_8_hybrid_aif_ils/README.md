# Figure 8: Hybrid AIF Instruction Locality Score

This folder contains the code used to recreate Figure 8 of the paper.

Figure 8 evaluates the Instruction Locality Score (ILS) of the Hybrid AIF framework across interaction steps.

- Figure 8(a): Hybrid AIF ILS on the small 5-object subset.
- Figure 8(b): Hybrid AIF ILS on the 200-object subset.

The scripts compute CLIP-based ILS values and plot sorted object-rank curves across interaction steps.

## Scripts

- `compute_ils_hybrid_aif.py`: computes the Hybrid AIF ILS CSV from generated base and edited outputs.
- `plot_hybrid_aif_ils_sorted_biglegend.py`: plots the sorted Hybrid AIF ILS curves.

## Recreate Figure 8(a)

From the MVDream working directory:

```bash
cd ~/MVDream
source .venv_ils/bin/activate

python compute_ils_hybrid_aif.py \
  --base_dir outputs_vis_aif_hybrid \
  --edit_dir outputs_vis_aif_hybrid_edit \
  --out_csv hybrid_aif_ils.csv

python plot_hybrid_aif_ils_sorted_biglegend.py \
  --csv hybrid_aif_ils.csv \
  --out_prefix commag_fig_hybrid_aif_ils_sorted
Recreate Figure 8(b)
cd ~/MVDream
source .venv_ils/bin/activate

python compute_ils_hybrid_aif.py \
  --base_dir outputs_vis_aif_hybrid_200 \
  --edit_dir outputs_vis_aif_hybrid_edit_200 \
  --out_csv hybrid_aif_ils_200.csv

python plot_hybrid_aif_ils_sorted_biglegend.py \
  --csv hybrid_aif_ils_200.csv \
  --out_prefix commag_fig_hybrid_aif_ils_sorted_200_biglegend_inside
Dependency

If open_clip is missing, install:

python -m pip install open_clip_torch


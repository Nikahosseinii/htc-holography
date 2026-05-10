# Figure 10: Prediction-Aware Update Reliability

This folder contains the code used to recreate Figure 10 of the paper.

Figure 10 evaluates prediction-aware update reliability under motion-driven viewpoint changes using:

- PAUE: Prediction-Aware Update Error
- PAFUE: Prediction-Aware Final Update Error

The comparison is between:

- Fixed Factorized
- Hybrid AIF

Lower PAUE and PAFUE indicate better future update reliability.

## Scripts

- `compare_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50.py`: runs the predictive Hybrid AIF comparison and creates `compare_metrics.csv`.
- `compute_paue_factorized_vs_hybrid.py`: computes PAUE/PAFUE summaries and plots the figures.

## Recreate Figure 10

From the MVDream working directory:

```bash
cd ~/MVDream
source .venv310/bin/activate

python compare_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50.py

python compute_paue_factorized_vs_hybrid.py \
  --csv outputs_compare_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50/compare_metrics.csv \
  --out_dir metrics_paue_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50 \
  --max_horizon 4

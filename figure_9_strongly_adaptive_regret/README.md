# Figure 9: Strongly Adaptive Regret

This folder contains the code used to recreate Figure 9 of the paper.

Figure 9 evaluates strongly adaptive regret for the interactive holographic update policy. It compares:

- Fixed Factorized
- Hybrid AIF

The figure reports regret over different interval lengths.

## Scripts

- `compare_factorized_vs_hybrid_regret_tuned_v3.py`: runs the tuned Fixed Factorized vs Hybrid AIF rollout and saves comparison metrics.
- `compare_adaptive_regret_factorized_vs_aif.py`: computes strongly adaptive regret from the rollout metrics and generates the final regret figure.

## Recreate Figure 9

From the MVDream working directory:

```bash
cd ~/MVDream
source .venv310/bin/activate

python compare_factorized_vs_hybrid_regret_tuned_v3.py
python compare_adaptive_regret_factorized_vs_aif.py

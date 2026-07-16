# Figure 14: TOPIQ-FR Comparison

This folder contains the plotting script for the paired event-level TOPIQ-FR comparison between Fixed Factorized transmission and Hybrid AIF.

## Script

topiq_factorized_vs_aif_cleanplot.py

## How to run

From the MVDream project folder, run:

    cd ~/MVDream

    python topiq_factorized_vs_aif_cleanplot.py \
      --run-dir outputs_compare_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50 \
      --out-dir outputs_topiq_comparison \
      --device cuda \
      --batch-size 1 \
      --bootstrap-samples 10000 \
      --permutation-samples 100000

## Expected output

The script recreates the TOPIQ-FR paired comparison figure in:

    outputs_topiq_comparison/

Expected output:

    topiq_fr_factorized_vs_hybrid_aif_clean.png

# SDOP Richer By-Action Curve

This folder contains the plotting script for the Semantic Distortion Outage Probability (SDOP) richer by-action curve.

## Script

plot_sdop_richer_by_action.py

## Required input

The script uses:

    outputs_sdop_aif_vs_factorized/sdop_per_event.csv

## How to run

From the MVDream project folder, run:

    cd ~/MVDream

    python plot_sdop_richer_by_action.py \
      --input-csv outputs_sdop_aif_vs_factorized/sdop_per_event.csv \
      --out-dir outputs_sdop_aif_vs_factorized \
      --primary-threshold 0.05 \
      --threshold-min 0.0 \
      --threshold-max 0.20 \
      --threshold-points 81 \
      --dpi 300

## Expected output

The script recreates the SDOP richer by-action outputs in:

    outputs_sdop_aif_vs_factorized/

Expected outputs:

    sdop_curve_richer_by_action.png
    sdop_curve_richer_by_action.pdf
    sdop_curve_richer_by_action.csv

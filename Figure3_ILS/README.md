# Figure 3: Instruction Locality Score

This folder contains the code and data used to reproduce Figure 3.

## Script

    plot_ils_sorted_richer_bigfont_bwfriendly.py

## Input data

    ils_baseline_vs_factorized_fixed_200.csv

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Reproduce Figure 3

Run:

    python ~/htc-holography/Figure3_ILS/plot_ils_sorted_richer_bigfont_bwfriendly.py

This creates:

    commag_fig_sorted_richer_bigfont_bwfriendly.pdf
    commag_fig_sorted_richer_bigfont_bwfriendly.svg

## Notes

The figure compares baseline and factorized-fixed Instruction Locality Score (ILS) curves across sorted object samples and different edit-strength values α.

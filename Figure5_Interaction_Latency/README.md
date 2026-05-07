# Figure 5: Interaction Latency

This folder contains the code and latency profile inputs used to reproduce Figure 5.

## Script

    plot_il_from_profiles_bigfont.py

## Input profile files

    baseline_50_prof.json
    fact_25_prof.json
    fact_10_prof.json

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Reproduce Figure 5

Run:

    python ~/htc-holography/Figure5_Interaction_Latency/plot_il_from_profiles_bigfont.py \
      --profiles ~/htc-holography/Figure5_Interaction_Latency/baseline_50_prof.json \
                 ~/htc-holography/Figure5_Interaction_Latency/fact_25_prof.json \
                 ~/htc-holography/Figure5_Interaction_Latency/fact_10_prof.json \
      --labels "Baseline (50 DDIM)" "Factorized (25 DDIM)" "Factorized (10 DDIM)" \
      --out IL_full_rho_sweep_bigfont.png \
      --R_mbps 50 100 300 1000 3000 10000 \
      --rhos 0.0 0.3 0.6 0.8

This creates:

    IL_full_rho_sweep_bigfont.png

## Notes

The figure shows end-to-end interaction latency as a function of link rate under different utilization values rho.

# Interactive Bandwidth Reduction Evaluation

This folder contains the code used to reproduce the payload and Interactive Bandwidth Reduction (IBR) results used for Table I.

IBR is computed from the communication payload model, not from generated images.

## Scripts

    payload_plot.py
    cumulative_plot_richer_bigfont.py

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Step 1: Payload-size breakdown

Run:

    python ~/htc-holography/IBR/payload_plot.py

This creates:

    payload_size_breakdown.png
    payload_size_breakdown.pdf

The figure shows the payload values used in the IBR calculation:

    |z_shared| = 946,176 bytes
    |z_view|   = 256 bytes
    |RGB|      = 512 * 512 * 3 = 786,432 bytes/frame
    |x0|       = 4 * 64 * 64 * 2 = 32,768 bytes/frame

## Step 2: Cumulative payload comparison

Run:

    python ~/htc-holography/IBR/cumulative_plot_richer_bigfont.py

This creates:

    cumulative_payload_comparison_richer_bigfont.png
    cumulative_payload_comparison_richer_bigfont.pdf

The script compares cumulative transmitted data for:

    RGB streaming
    latent streaming
    proposed factorized transmission

## How the Table I IBR number is obtained

The bandwidths are computed as:

    B_RGB = FPS * |RGB|

    B_latent = FPS * |x0|

    B_factorized = |z_shared| / T + FPS * |z_view|

Using:

    FPS = 30
    T = 60 seconds

The IBR is:

    IBR = B_stream / B_factorized

This gives approximately:

    IBR vs RGB streaming ≈ 1026x
    IBR vs latent streaming ≈ 42.6x

Therefore, the Table I entry is reported as:

    Baseline MVDream: 1x
    Factorized: 42x-1026x

# Figure 4: Cumulative transmitted data comparison

This folder contains the code used to reproduce Figure 4.

## Script

    cumulative_plot_richer_bwfriendly.py

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Reproduce Figure 4

Run:

    python ~/htc-holography/Figure4_IBR/cumulative_plot_richer_bwfriendly.py

This creates the cumulative transmitted data figure comparing:

    RGB streaming
    latent streaming
    proposed factorized transmission

## Notes

The script produces the black-and-white friendly version of the cumulative payload comparison figure used in the paper.

The plot is based on the following payload assumptions:

    |z_shared| = 946,176 bytes
    |z_view|   = 256 bytes
    |RGB|      = 512 * 512 * 3 = 786,432 bytes/frame
    |x0|       = 4 * 64 * 64 * 2 = 32,768 bytes/frame

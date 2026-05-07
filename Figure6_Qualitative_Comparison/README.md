# Figure 6: Qualitative Comparison

This folder contains the code used to reproduce Figure 6.

## Script

    make_fig6_consistent.py

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Reproduce Figure 6

Run:

    python ~/htc-holography/Figure6_Qualitative_Comparison/make_fig6_consistent.py

This creates:

    fig6_consistent.png

## Notes

Figure 6 compares qualitative multi-view outputs for the same selected objects across:

    Baseline (50 DDIM)
    Factorized (25 DDIM)
    Factorized (10 DDIM)

The selected object IDs used by the script are:

    1aa6f55db8ad43c48e6683efcb596847
    480b81bd14f143aa83c54aeb5ae91112
    5b139cf307c54e5690795b68a3df0ef6

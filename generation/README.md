# Generation Scripts

This folder contains the baseline and factorized generation scripts used to generate the image outputs for Table I and related experiments.

## Included scripts

### Baseline

    baseline/gen_val_200_v21_resume_baseline.py

### Factorized

    factorized/gen_val_200_v21_resume_factorized.py

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Baseline generation

    python ~/htc-holography/generation/baseline/gen_val_200_v21_resume_baseline.py

## Factorized generation

    python ~/htc-holography/generation/factorized/gen_val_200_v21_resume_factorized.py

## Notes

These scripts regenerate the baseline and factorized image outputs. The generated image folders themselves are not included in this repository because they are large.

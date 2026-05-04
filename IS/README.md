# Inception Score Evaluation

This folder contains the script used to reproduce the Inception Score (IS) results reported in Table I.

## Usage

From the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

Run IS for the baseline/original MVDream outputs:

    python ~/htc-holography/IS/eval_inception_score.py \
      --img_root ~/MVDream/generated_v21_objaverse

Expected baseline result:

    Images scored: 4000
    Inception Score: mean approximately 9.49, std approximately 0.36

Run IS for the factorized outputs by replacing --img_root with the factorized output folder, for example:

    python ~/htc-holography/IS/eval_inception_score.py \
      --img_root ~/MVDream/generated_v21_objaverse_factorized

The Table I factorized result should be close to:

    Inception Score: mean approximately 4.16, std approximately 0.20

## Notes

The script evaluates generated image folders directly. It does not regenerate images or figures.

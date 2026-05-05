# Inception Score Evaluation

This folder contains the script used to reproduce the Inception Score (IS) results reported in Table I.

The script evaluates already-generated image folders. It does not regenerate images, figures, or tables.

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Baseline MVDream IS

Use the full 4000-image baseline validation folder:

    python ~/htc-holography/IS/eval_inception_score.py \
      --img_root ~/MVDream/generated_v21_objaverse

Expected result:

    Images scored: 4000
    Inception Score: mean approximately 9.49

## Factorized IS

Use the full 4000-image factorized validation folder:

    python ~/htc-holography/IS/eval_inception_score.py \
      --img_root ~/MVDream/generated_v21_objaverse_factorized

Expected result:

    Images scored: 4000
    Inception Score: mean approximately 4.16

## Notes

The small folders `outputs_vis_factorized10` and `outputs_vis_factorized25` contain only visualization samples and are not used for the Table I IS evaluation.

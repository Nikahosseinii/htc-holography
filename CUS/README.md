# Camera-Update Stability Evaluation

This folder contains the CUS evaluation script used to reproduce the Camera-Update Stability column of Table I.

CUS measures view-to-view stability by computing CLIP-image feature similarity across generated views of the same object.

## Script

    eval_cus_clip.py

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Baseline MVDream CUS

Run:

    python ~/htc-holography/CUS/eval_cus_clip.py \
      --img_root ~/MVDream/generated_v21_objaverse

Expected Table I value:

    CUS approximately 0.908

## Factorized CUS

Run the same script with the factorized folder:

    python ~/htc-holography/CUS/eval_cus_clip.py \
      --img_root ~/MVDream/generated_v21_objaverse_factorized

Expected Table I value:

    CUS approximately 0.999

## Notes

The same script is used for both baseline and factorized outputs. The only difference is the `--img_root` folder.

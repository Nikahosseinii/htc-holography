# CLIP Score Evaluation

This folder contains the CLIP evaluation script used to reproduce the CLIP column of Table I.

The script evaluates already-generated image folders. It does not regenerate images, figures, or tables.

## Script

    eval_clipscore.py

## Environment

Run from the MVDream environment:

    cd ~/MVDream
    source .venv310/bin/activate

## Baseline MVDream CLIP

Run:

    python ~/htc-holography/CLIP/eval_clipscore.py \
      --img_root ~/MVDream/generated_v21_objaverse \
      --prompts_json ~/MVDream/val_prompts_1000_objaverse.json

Expected Table I value:

    CLIP approximately 26.46

## Factorized CLIP

Run the same script, but change the image folder to the factorized outputs:

    python ~/htc-holography/CLIP/eval_clipscore.py \
      --img_root ~/MVDream/generated_v21_objaverse_factorized \
      --prompts_json ~/MVDream/val_prompts_1000_objaverse.json

Expected Table I value:

    CLIP approximately 25.41

## Notes

The same script is used for both baseline and factorized outputs. The only difference is the `--img_root` folder.

The paper reports CLIP using the multiplied-by-100 convention. If the script prints raw cosine similarity, multiply the raw value by 100.

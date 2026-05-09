import os
import re
import csv
import argparse
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip


PNG_RE = re.compile(r"^(?P<obj>.+)_step_(?P<step>\d+)_view_(?P<view>\d+)\.png$")


def parse_name(fn: str):
    m = PNG_RE.match(fn)
    if not m:
        return None
    return (
        m.group("obj"),
        int(m.group("step")),
        int(m.group("view")),
    )


def index_dir(root: str) -> Dict[Tuple[str, int, int], str]:
    out = {}
    for fn in os.listdir(root):
        if not fn.lower().endswith(".png"):
            continue
        parsed = parse_name(fn)
        if parsed is None:
            continue
        out[parsed] = os.path.join(root, fn)
    return out


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def encode_images(model, preprocess, device: str, paths: List[str]) -> torch.Tensor:
    imgs = [preprocess(load_image(p)) for p in paths]
    x = torch.stack(imgs, dim=0).to(device)
    feats = model.encode_image(x)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu()


def cosine_1m(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="Hybrid AIF outputs for original/base prompt")
    ap.add_argument("--edit_dir", required=True,
                    help="Hybrid AIF outputs for edited prompt")
    ap.add_argument("--out_csv", default="hybrid_aif_ils.csv")
    ap.add_argument("--view_count", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base_index = index_dir(args.base_dir)
    edit_index = index_dir(args.edit_dir)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    clip_model = clip_model.eval().to(args.device)

    # collect common (obj, step)
    base_keys = {(obj, step) for (obj, step, view) in base_index.keys()}
    edit_keys = {(obj, step) for (obj, step, view) in edit_index.keys()}
    common = sorted(base_keys & edit_keys)

    rows = []

    for obj_id, step in common:
        base_paths = []
        edit_paths = []

        missing = False
        for v in range(args.view_count):
            kb = (obj_id, step, v)
            ke = (obj_id, step, v)
            if kb not in base_index or ke not in edit_index:
                missing = True
                break
            base_paths.append(base_index[kb])
            edit_paths.append(edit_index[ke])

        if missing:
            continue

        feats = encode_images(
            clip_model, preprocess, args.device, base_paths + edit_paths
        )
        base_feats = feats[:args.view_count]
        edit_feats = feats[args.view_count:]

        delta_cam = cosine_1m(base_feats[0:1], base_feats[1:2])[0].item()
        delta_instr = cosine_1m(base_feats, edit_feats)
        delta_instr_mean = delta_instr.mean().item()

        ils = delta_instr_mean / (delta_cam + args.eps)

        rows.append({
            "id": obj_id,
            "step": step,
            "model": "hybrid_aif",
            "ils": float(ils),
            "delta_cam": float(delta_cam),
            "delta_instr_mean": float(delta_instr_mean),
        })

    if not rows:
        raise ValueError("No matched object-step-view sets found between base_dir and edit_dir.")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {args.out_csv} with {len(rows)} rows.")


if __name__ == "__main__":
    main()

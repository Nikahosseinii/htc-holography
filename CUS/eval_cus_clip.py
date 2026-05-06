import os, re, argparse
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
import open_clip
from tqdm import tqdm

EXTS = (".png", ".jpg", ".jpeg", ".webp")
PAT = re.compile(r"^([0-9a-fA-F]+)_view_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

def group_images(img_root):
    groups = defaultdict(dict)  # id -> {view_idx: path}
    for f in os.listdir(img_root):
        if not f.lower().endswith(EXTS):
            continue
        m = PAT.match(f)
        if not m:
            continue
        obj_id = m.group(1)
        view_i = int(m.group(2))
        groups[obj_id][view_i] = os.path.join(img_root, f)
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--views", type=str, default="0,1,2,3",
                    help="Comma-separated view indices to use (must be consecutive for CUS). Default: 0,1,2,3")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    view_list = [int(x.strip()) for x in args.views.split(",") if x.strip() != ""]
    view_list = sorted(view_list)

    # CLIP image encoder
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    groups = group_images(args.img_root)
    if not groups:
        raise RuntimeError("No files matched pattern <id>_view_XX.png in img_root.")

    per_obj_cos = []
    skipped = 0

    with torch.no_grad():
        for obj_id in tqdm(sorted(groups.keys())):
            view_map = groups[obj_id]
            # require all specified views
            if any(v not in view_map for v in view_list):
                skipped += 1
                continue

            # encode all requested views
            feats = {}
            for v in view_list:
                img = Image.open(view_map[v]).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats[v] = f.squeeze(0)

            # consecutive cosine similarities
            cos_sims = []
            for i in range(len(view_list) - 1):
                v1, v2 = view_list[i], view_list[i+1]
                cos = float((feats[v1] * feats[v2]).sum().item())
                cos_sims.append(cos)

            per_obj_cos.append(float(np.mean(cos_sims)))

    per_obj_cos = np.array(per_obj_cos, dtype=np.float32)

    # CUS = E[cos] (since drift = 1-cos and CUS = 1-E[drift])
    cus_mean = float(per_obj_cos.mean())
    cus_std  = float(per_obj_cos.std())

    print(f"\nObjects found: {len(groups)}")
    print(f"Objects scored: {len(per_obj_cos)} | Skipped (missing views): {skipped}")
    print(f"CUS (mean cosine across consecutive views): mean={cus_mean:.4f}, std={cus_std:.4f}")
    print(f"Median={float(np.median(per_obj_cos)):.4f}, 5%={float(np.percentile(per_obj_cos,5)):.4f}, 95%={float(np.percentile(per_obj_cos,95)):.4f}\n")

if __name__ == "__main__":
    main()

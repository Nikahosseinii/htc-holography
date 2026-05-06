import os, json, argparse, re
from PIL import Image
import torch
import open_clip
from tqdm import tqdm
import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".webp")
ID_RE = re.compile(r"^([0-9a-fA-F]+)_view_\d+")

def list_images(root):
    imgs = []
    for f in os.listdir(root):
        if f.lower().endswith(EXTS):
            imgs.append(os.path.join(root, f))
    return sorted(imgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--prompts_json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    data = json.load(open(args.prompts_json))
    # JSON is a list of dicts: {"id": ..., "prompt": ...}
    prompt_map = {x["id"]: x["prompt"] for x in data}

    imgs = list_images(args.img_root)
    if not imgs:
        raise RuntimeError("No images found in img_root.")

    scores = []
    missing = 0

    with torch.no_grad():
        for path in tqdm(imgs):
            name = os.path.basename(path)
            m = ID_RE.match(name)
            img_id = m.group(1) if m else name.split("_")[0]
            prompt = prompt_map.get(img_id)
            if prompt is None:
                missing += 1
                continue

            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)

            img_f = model.encode_image(image)
            txt_f = model.encode_text(text)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

            scores.append((img_f * txt_f).sum(dim=-1).item())

    scores = np.array(scores, dtype=np.float32)
    print(f"\nTotal images: {len(imgs)}")
    print(f"Matched prompts: {len(scores)} | Missing prompts: {missing}")

    if len(scores) == 0:
        raise RuntimeError("Matched 0 images unexpectedly. Something is wrong with filenames or prompts map.")

    print(f"CLIP cosine similarity: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"Median={np.median(scores):.4f}, 5%={np.percentile(scores,5):.4f}, 95%={np.percentile(scores,95):.4f}\n")

if __name__ == "__main__":
    main()

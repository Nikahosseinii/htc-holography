import os, argparse
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore

EXTS = (".png", ".jpg", ".jpeg", ".webp")

def list_images(root):
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.lower().endswith(EXTS)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_images", type=int, default=5000)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    imgs = list_images(args.img_root)[:args.max_images]
    if not imgs:
        raise RuntimeError("No images found.")

    # Inception v3 expects 299×299
    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    metric = InceptionScore(
        splits=10,
        normalize=True
    ).to(device)

    with torch.no_grad():
        for p in tqdm(imgs):
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            metric.update(x)

    mean, std = metric.compute()
    print(f"\nImages scored: {len(imgs)}")
    print(f"Inception Score: mean={mean.item():.4f}, std={std.item():.4f}\n")

if __name__ == "__main__":
    main()

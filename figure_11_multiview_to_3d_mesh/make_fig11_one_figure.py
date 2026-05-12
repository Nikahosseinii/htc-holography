from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

base = Path("/mnt/d/University/WINTER26/Report12")

# Top row: 2D conditioning views from MVDream
cond_files = [
    base / "object1_views" / "back.png",
    base / "object1_views" / "left.png",
    base / "object1_views" / "right.png",
    base / "object1_views" / "views.png",
]
cond_labels = ["Back", "Left", "Right", "View"]

# Bottom rows: 8 rendered white-mesh views
mesh_files = [
    base / "front.jpg",
    base / "left.jpg",
    base / "right.jpg",
    base / "back.jpg",
    base / "top.jpg",
    base / "bottom.jpg",
    base / "tilted1.jpg",
    base / "tilted2.jpg",
]
mesh_labels = ["Front", "Left", "Right", "Back", "Top", "Bottom", "Tilted 1", "Tilted 2"]

# Check all required files exist
for p in cond_files + mesh_files:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

cond_imgs = [Image.open(p).convert("RGB") for p in cond_files]
mesh_imgs = [Image.open(p).convert("RGB") for p in mesh_files]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
})

fig, axes = plt.subplots(
    3,
    4,
    figsize=(14, 12),
    gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
)

# Top row: conditioning views
for j in range(4):
    ax = axes[0, j]
    ax.imshow(cond_imgs[j])
    ax.set_title(cond_labels[j], fontsize=15, pad=6, fontweight="bold")
    ax.axis("off")

# Middle and bottom rows: mesh views
for idx in range(8):
    r = 1 + idx // 4
    c = idx % 4
    ax = axes[r, c]
    ax.imshow(mesh_imgs[idx])
    ax.set_title(mesh_labels[idx], fontsize=15, pad=6, fontweight="bold")
    ax.axis("off")

# Section labels
fig.text(
    0.5,
    0.975,
    "(a) Factorized MVDream multi-view renders used as conditioning views for Hunyuan3D-2mv",
    ha="center",
    va="top",
    fontsize=17,
    fontweight="bold",
)

fig.text(
    0.5,
    0.655,
    "(b) Corresponding white mesh shown from eight viewpoints",
    ha="center",
    va="top",
    fontsize=17,
    fontweight="bold",
)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95], h_pad=1.6, w_pad=0.8)

out_path = base / "fig11_one_figure_bigfont.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Saved: {out_path}")

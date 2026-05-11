from pathlib import Path
from PIL import Image, ImageDraw

# Input/output folders
in_dir = Path("/mnt/d/aif_results/outputs_vis_aif_hybrid")
out_dir = Path("/mnt/d/aif_results")
out_dir.mkdir(parents=True, exist_ok=True)

# Use one object from the generated Hybrid AIF outputs.
# This object appears in your folder list.
obj_id = "1aa6f55db8ad43c48e6683efcb596847"

# Four interaction steps to show as columns.
# Change this if you want different columns.
steps = [0, 2, 4, 6]

# Four views to show as rows.
views = [0, 1, 2, 3]

cell_size = 256
label_h = 34
pad = 10

cols = len(steps)
rows = len(views)

canvas_w = cols * cell_size + (cols + 1) * pad
canvas_h = rows * cell_size + (rows + 1) * pad + label_h

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)

# Column labels
for c, step in enumerate(steps):
    x = pad + c * (cell_size + pad)
    draw.text((x + 8, 8), f"step {step}", fill=(0, 0, 0))

# Paste images
for r, view in enumerate(views):
    for c, step in enumerate(steps):
        img_path = in_dir / f"{obj_id}_step_{step:02d}_view_{view:02d}.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB").resize((cell_size, cell_size))

        x = pad + c * (cell_size + pad)
        y = label_h + pad + r * (cell_size + pad)

        canvas.paste(img, (x, y))

out_path = out_dir / "aif_paper_figure_all4views_recreated.png"
canvas.save(out_path)

print(f"Saved: {out_path}")

from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def parse_name(p: Path):
    # expected pattern: <object_id>_view_00.png
    name = p.stem
    if "_view_" not in name:
        return None, None
    obj_id, view_id = name.rsplit("_view_", 1)
    return obj_id, view_id


def build_index(folder: Path):
    idx = {}
    for p in sorted(folder.glob("*.png")):
        obj_id, view_id = parse_name(p)
        if obj_id is None:
            continue
        idx.setdefault(obj_id, {})[view_id] = p
    return idx


def choose_common_objects(idx_a, idx_b, idx_c, required_views=("00", "01", "02", "03")):
    common = sorted(set(idx_a) & set(idx_b) & set(idx_c))
    good = []
    for obj_id in common:
        ok = all(v in idx_a[obj_id] and v in idx_b[obj_id] and v in idx_c[obj_id] for v in required_views)
        if ok:
            good.append(obj_id)
    return good


def make_panel_image(paths_for_one_model, tile_size=(220, 220)):
    views = []
    for p in paths_for_one_model:
        img = Image.open(p).convert("RGB")
        img = img.resize(tile_size)
        views.append(img)

    w, h = tile_size
    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    canvas.paste(views[0], (0, 0))
    canvas.paste(views[1], (w, 0))
    canvas.paste(views[2], (0, h))
    canvas.paste(views[3], (w, h))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default="outputs_vis_baseline50")
    ap.add_argument("--fact25_dir", default="outputs_vis_factorized25")
    ap.add_argument("--fact10_dir", default="outputs_vis_factorized10")
    ap.add_argument("--out", default="fig6_consistent.png")
    ap.add_argument("--rows", type=int, default=3, help="number of object rows to show")
    ap.add_argument("--object_ids", nargs="*", default=None,
                    help="optional explicit object IDs to use, in order")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    fact25_dir = Path(args.fact25_dir)
    fact10_dir = Path(args.fact10_dir)

    idx_b = build_index(baseline_dir)
    idx_f25 = build_index(fact25_dir)
    idx_f10 = build_index(fact10_dir)

    common_objects = choose_common_objects(idx_b, idx_f25, idx_f10)

    if not common_objects:
        raise RuntimeError("No common object IDs with all 4 views found across the three folders.")

    if args.object_ids:
        selected = [x for x in args.object_ids if x in common_objects]
        if len(selected) == 0:
            raise RuntimeError("None of the requested object IDs exist in all three folders with all 4 views.")
    else:
        selected = common_objects[:args.rows]

    selected = selected[:args.rows]

    col_titles = ["Baseline (50 DDIM)", "Factorized (25 DDIM)", "Factorized (10 DDIM)"]
    required_views = ["00", "01", "02", "03"]

    fig, axes = plt.subplots(len(selected), 3, figsize=(11, 3.6 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for r, obj_id in enumerate(selected):
        paths_b = [idx_b[obj_id][v] for v in required_views]
        paths_f25 = [idx_f25[obj_id][v] for v in required_views]
        paths_f10 = [idx_f10[obj_id][v] for v in required_views]

        panels = [
            make_panel_image(paths_b),
            make_panel_image(paths_f25),
            make_panel_image(paths_f10),
        ]

        for c in range(3):
            ax = axes[r][c]
            ax.imshow(panels[c])
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=13, pad=10, fontweight="bold")

        # row label on the left
        axes[r][0].text(
            -0.08, 0.5, f"Object {r+1}",
            transform=axes[r][0].transAxes,
            va="center", ha="right",
            fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {args.out}")
    print("[OK] Objects used:", selected)


if __name__ == "__main__":
    main()

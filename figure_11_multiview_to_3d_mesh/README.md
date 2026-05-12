# Figure 11: From MVDream Multi-View Outputs to 3D Mesh

This folder contains the code used to recreate Figure 11 of the paper.

Figure 11 demonstrates the bridge from 2D multi-view generative outputs to a 3D mesh representation. The workflow uses four MVDream-rendered views as conditioning inputs for Hunyuan3D-2mv, generates a 3D mesh, and then assembles the paper figure showing both the input views and rendered mesh views.

## Scripts

### `run_2mv.py`

This script runs Hunyuan3D-2mv on four MVDream output views.

Input:

```text
/home/nika/hunyuan_mv_inputs/object1/front.png
/home/nika/hunyuan_mv_inputs/object1/left.png
/home/nika/hunyuan_mv_inputs/object1/back.png
/home/nika/hunyuan_mv_inputs/object1/right.png

Output:

/home/nika/hunyuan_mv_inputs/object1/object1_2mv.glb

This is the generated 3D mesh file.

make_fig11_one_figure.py

This script creates the final Figure 11 image by combining:

the four MVDream conditioning views, and
eight rendered views of the generated white mesh.

Input conditioning views:

/mnt/d/University/WINTER26/Report12/object1_views/back.png
/mnt/d/University/WINTER26/Report12/object1_views/left.png
/mnt/d/University/WINTER26/Report12/object1_views/right.png
/mnt/d/University/WINTER26/Report12/object1_views/views.png

Input mesh-view renders:

/mnt/d/University/WINTER26/Report12/front.jpg
/mnt/d/University/WINTER26/Report12/left.jpg
/mnt/d/University/WINTER26/Report12/right.jpg
/mnt/d/University/WINTER26/Report12/back.jpg
/mnt/d/University/WINTER26/Report12/top.jpg
/mnt/d/University/WINTER26/Report12/bottom.jpg
/mnt/d/University/WINTER26/Report12/tilted1.jpg
/mnt/d/University/WINTER26/Report12/tilted2.jpg

Output:

/mnt/d/University/WINTER26/Report12/fig11_one_figure_bigfont.png
Recreate Figure 11
Step 1: Generate the 3D mesh
source ~/.venv_hy3d/bin/activate
cd ~/Hunyuan3D-2

python run_2mv.py
Step 2: Create the final figure
cd ~/MVDream
source .venv310/bin/activate

python make_fig11_one_figure.py
Copy Output to E Drive
mkdir -p "/mnt/e/figure 11"

cp /mnt/d/University/WINTER26/Report12/fig11_one_figure_bigfont.png "/mnt/e/figure 11/"

ls -lh "/mnt/e/figure 11"
Notes
run_2mv.py creates the 3D mesh from four MVDream views.
make_fig11_one_figure.py does not generate the mesh. It only assembles the final paper figure from existing input views and rendered mesh views.

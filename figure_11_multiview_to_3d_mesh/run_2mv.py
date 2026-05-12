import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv",
    use_safetensors=True,
    device="cuda"
)

mesh = pipe(
    image={
        "front": "/home/nika/hunyuan_mv_inputs/object1/front.png",
        "left": "/home/nika/hunyuan_mv_inputs/object1/left.png",
        "back": "/home/nika/hunyuan_mv_inputs/object1/back.png",
        "right": "/home/nika/hunyuan_mv_inputs/object1/right.png",
    },
    num_inference_steps=30,
    octree_resolution=256,
    num_chunks=10000,
    generator=torch.manual_seed(1234),
    output_type="trimesh",
)[0]

mesh.export("/home/nika/hunyuan_mv_inputs/object1/object1_2mv.glb")
print("saved: /home/nika/hunyuan_mv_inputs/object1/object1_2mv.glb")

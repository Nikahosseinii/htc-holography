import os
import json
import gc
import random
import numpy as np
import torch
from PIL import Image
import csv
# ---------------- CONFIG ----------------
VAL_JSON = "val_prompts_5.json"
OUT_DIR = "outputs_vis_baseline50"
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving outputs to:", OUT_DIR)

MODEL_NAME = "sd-v1.5-4view"
NUM_FRAMES = 4
SIZE = 256
DEVICE = "cuda"
SEED = 23
LATENCY_CSV = "latency_proc_baseline.csv"
WARMUP_ITEMS = 0
DEBUG = False
# ----------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def already_done(item_id):
    return all(
        os.path.exists(os.path.join(OUT_DIR, f"{item_id}_view_{i:02d}.png"))
        for i in range(NUM_FRAMES)
    )

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8,
        ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt) != list:
        prompt = [prompt]
    # Make sure CLIP is on GPU
    model.cond_stage_model = model.cond_stage_model.to(device)

    with torch.no_grad():
        print("CLIP param device:", next(model.cond_stage_model.parameters()).device)
        c = model.cond_stage_model(prompt)
        # Make context match the effective batch = batch_size * num_frames
        eff_bs = batch_size * num_frames
        c_  = {"context": c.repeat(eff_bs, 1, 1)}
        uc_ = {"context": uc.repeat(eff_bs, 1, 1)}

        if camera is not None:
            if hasattr(camera, "shape") and camera.shape[0] != eff_bs:
                assert eff_bs % camera.shape[0] == 0, f"camera batch {camera.shape[0]} doesn't divide eff_bs {eff_bs}"
                camera = camera.repeat(eff_bs // camera.shape[0], 1)

            c_["camera"] = uc_["camera"] = camera

            c_["num_frames"] = uc_["num_frames"] = int(num_frames)


        shape = [4, image_size // 8, image_size // 8]
        print("COND KEYS:", c_.keys())
        def _bytes_of(v):
            import torch
            if v is None:
                return 0
            if torch.is_tensor(v):
                return v.numel() * v.element_size()
            if isinstance(v, (list, tuple)):
                return sum(_bytes_of(x) for x in v)
            if isinstance(v, dict):
                return sum(_bytes_of(x) for x in v.values())
            return 0

        if DEBUG:
            print("\n=== PAYLOAD DEBUG ===")
            print("COND KEYS:", list(c_.keys()))
            for k, v in c_.items():
                print(f"{k:>12s} bytes: {_bytes_of(v)}")
            print("=====================\n") 
            print("DEBUG shapes:")
            print("  batch_size:", batch_size, "num_frames:", num_frames)
            print("  cond context:", c_["context"].shape)
            print("  uncond context:", uc_["context"].shape)
        if "camera" in c_:
            cam = c_["camera"]
            print("  camera type:", type(cam))
            try:
                print("  camera shape:", cam.shape)
            except:
                print("  camera:", cam)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.cuda.synchronize()
        ev_s0 = torch.cuda.Event(enable_timing=True)
        ev_s1 = torch.cuda.Event(enable_timing=True)
        ev_s0.record()

        samples_ddim, _ = sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=eff_bs,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None
        )
        ev_s1.record()
        torch.cuda.synchronize()
        sample_ms = float(ev_s0.elapsed_time(ev_s1))
        torch.cuda.synchronize()
        ev_d0 = torch.cuda.Event(enable_timing=True)
        ev_d1 = torch.cuda.Event(enable_timing=True)
        ev_d0.record()

        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()
        x_sample = x_sample.astype(np.uint8)
        ev_d1.record()
        torch.cuda.synchronize()
        decode_ms = float(ev_d0.elapsed_time(ev_d1))

# Ensure we return exactly num_frames views for ONE object
# Many pipelines produce B = batch_size * num_frames or B = batch_size.
# The safest assumption here: frames for the first item are the first num_frames images.
    frames = [x_sample[i] for i in range(min(num_frames, x_sample.shape[0]))]
    return frames, sample_ms, decode_ms


def time_gpu_ms(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out  # ms, return value
# --- IL hook: one update runner (factorized) ---------------------------------
_IL_STATE = {"ready": False}

def _il_init(ddim_steps: int):
    """
    Initialize once: model, sampler, camera, uc, etc.
    Mirrors your script's setup.
    """
    global _IL_STATE
    if _IL_STATE["ready"]:
        return

    from mvdream.camera_utils import get_camera
    from mvdream.ldm.models.diffusion.ddim import DDIMSampler
    from mvdream.model_zoo import build_model
    import torch

    DEVICE = torch.device("cuda:0")
    dtype = torch.float16

    model = build_model(MODEL_NAME, ckpt_path=None)
    model.factorize_eps = False

    model = model.to(DEVICE)
    model.device = DEVICE
    model.cond_stage_model = model.cond_stage_model.to(DEVICE)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.cond_stage_model([""])

    camera = get_camera(NUM_FRAMES, elevation=15, azimuth_start=90, azimuth_span=360)
    camera = camera.repeat(1, 1).to(DEVICE)  # batch_size=1

    _IL_STATE.update({
        "ready": True,
        "model": model,
        "sampler": sampler,
        "uc": uc,
        "camera": camera,
        "ddim_steps": ddim_steps,
        "dtype": dtype,
        "device": DEVICE,
    })

def run_one_update(ddim_steps: int = 50):
    """
    Runs one viewpoint-update generation and returns:
      payload_bits, sample_ms, decode_ms
    """
    import torch
    import numpy as np
    ddim_steps = int(ddim_steps)
    _il_init(ddim_steps)
    st = _IL_STATE
    model = st["model"]
    sampler = st["sampler"]
    uc = st["uc"]
    camera = st["camera"]
    DEVICE = st["device"]
    dtype = st["dtype"]

    # choose ONE prompt (or hardcode a representative one)
    prompt = "a 3d asset"

    # build conditioning like your t2i()
    with torch.no_grad():
        c = model.cond_stage_model([prompt])
        eff_bs = 1 * NUM_FRAMES
        c_ = {"context": c.repeat(eff_bs, 1, 1)}
        uc_ = {"context": uc.repeat(eff_bs, 1, 1)}
        c_["camera"] = camera.repeat(eff_bs // NUM_FRAMES, 1)
        uc_["camera"] = camera.repeat(eff_bs // NUM_FRAMES, 1)
        c_["num_frames"] = int(NUM_FRAMES)
        uc_["num_frames"] = int(NUM_FRAMES)

        shape = [4, SIZE // 8, SIZE // 8]

        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()

        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c_,
            batch_size=eff_bs,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=10,
            unconditional_conditioning=uc_,
            eta=0.0,
            x_T=None,
        )

        ev1.record()
        torch.cuda.synchronize()
        sample_ms = float(ev0.elapsed_time(ev1))

        torch.cuda.synchronize()
        ev2 = torch.cuda.Event(enable_timing=True)
        ev3 = torch.cuda.Event(enable_timing=True)
        ev2.record()

        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, 0.0, 1.0)

        ev3.record()
        torch.cuda.synchronize()
        decode_ms = float(ev2.elapsed_time(ev3))

    # ---- payload bits for the NEW delay model ----
    # For instruction-based scheme, B should be the transmitted update payload.
    # If you are sending z_view only, set it here. If you send full latent per-frame, change it.
    # Start with your paper's measured z_view = 256 bytes (example):
    rgb_bits_per_frame = SIZE * SIZE * 3 * 8
    payload_bits = int(NUM_FRAMES * rgb_bits_per_frame)

    return payload_bits, sample_ms, decode_ms

if __name__ == "__main__":
    from mvdream.camera_utils import get_camera
    from mvdream.ldm.models.diffusion.ddim import DDIMSampler
    from mvdream.model_zoo import build_model

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter to only items not yet done
    pending = [item for item in data if not already_done(item["id"])]
    #pending = pending[:2]
    print(f"Total: {len(data)}, Already done: {len(data) - len(pending)}, Pending: {len(pending)}")

    if not pending:
        print("All items already generated!")
        exit(0)

    # Load model ONCE
    # Load model ONCE
    print("Loading model...")
    dtype = torch.float16
    batch_size = max(4, NUM_FRAMES)

    DEVICE = torch.device("cuda:0")

    model = build_model(MODEL_NAME, ckpt_path=None)

# enable your custom logic
    model.factorize_eps = False

# move everything to GPU
    model = model.to(DEVICE)
    model.device = DEVICE                 # ✅ THIS IS THE KEY LINE
    model.cond_stage_model = model.cond_stage_model.to(DEVICE)

    model.eval()

    print("MODEL TYPE:", type(model))
    print("HAS apply_model:", hasattr(model, "apply_model"))
    print("HAS model attr:", hasattr(model, "model"))
    if hasattr(model, "model"):
        print("INNER TYPE:", type(model.model))
        print("INNER HAS apply_model:", hasattr(model.model, "apply_model"))

# optional hard stop if missing
    assert hasattr(model, "apply_model"), "Top-level model passed to DDIMSampler must have apply_model"

    sampler = DDIMSampler(model)

    sampler = DDIMSampler(model)
    uc = model.cond_stage_model([""])

    # Pre-compute camera matrices
    camera = get_camera(NUM_FRAMES, elevation=15, azimuth_start=90, azimuth_span=360)
    camera = camera.repeat(batch_size // NUM_FRAMES, 1).to(DEVICE)

    print(f"Model loaded. Processing {len(pending)} items...")
    timings = []      # rows: obj_id, proc_ms_total, proc_ms_per_view
    item_counter = 0


    # Process all pending items
    for idx, item in enumerate(pending, 1):
        item_id = item["id"]
        prompt = item["prompt"].strip()

        # Add suffix if needed
        if "3d asset" not in prompt.lower():
            prompt = prompt.rstrip(",") + ", 3d asset"

        print(f"[{idx}/{len(pending)}] Generating: {item_id}")
        set_seed(SEED)

        frames, sample_ms, decode_ms = t2i(
        model=model,
        image_size=SIZE,
        prompt=prompt,
        uc=uc,
        sampler=sampler,
        step=50,
        scale=10,
        batch_size=1,
        ddim_eta=0.0,
        dtype=dtype,
        device=DEVICE,
        camera=camera,
        num_frames=NUM_FRAMES
    )

        proc_ms = sample_ms + decode_ms


# warm-up to avoid CUDA init / caching effects
        if item_counter >= WARMUP_ITEMS:
            timings.append((
            item_id,
            proc_ms,
            proc_ms / float(NUM_FRAMES),
            sample_ms,
            decode_ms
        ))

        item_counter += 1
        print("DEBUG item_counter:", item_counter)

        img = np.concatenate(frames, axis=1)



        # Save individual views
        for i in range(NUM_FRAMES):
            view = img[:, i * SIZE:(i + 1) * SIZE, :]
            Image.fromarray(view).save(f"{OUT_DIR}/{item_id}_view_{i:02d}.png")

        # Cleanup between generations
            gc.collect()
            torch.cuda.empty_cache()
    with open(LATENCY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obj_id","proc_ms_total","proc_ms_per_view","sample_ms_total","decode_ms_total"])
        w.writerows(timings)
    print("Saved:", LATENCY_CSV, "rows:", len(timings))

    print(f"\nDone. Generated {len(pending)} items to ./{OUT_DIR}/")

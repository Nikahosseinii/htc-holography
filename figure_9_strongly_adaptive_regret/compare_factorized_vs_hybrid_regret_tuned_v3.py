#!/usr/bin/env python3
"""
compare_factorized_vs_hybrid_regret_tuned_v3.py

Full standalone comparison script.

Purpose:
- Compare a fixed factorized policy vs a loss-aligned hybrid AIF policy
- Use 10 DDIM steps everywhere for fairer interaction comparison
- Save per-step metrics for adaptive regret evaluation

Important:
This hybrid policy is aligned to the same kind of step loss used later by
compare_adaptive_regret_factorized_vs_aif.py:
    ref_error + payload_bits + proc_ms_total + recon_error_vs_prev

Outputs:
- outputs_compare_factorized_vs_hybrid_regret_tuned_v3/compare_metrics.csv
- outputs_compare_factorized_vs_hybrid_regret_tuned_v3/<obj_id>/{reference,factorized,hybrid}/...
- outputs_compare_factorized_vs_hybrid_regret_tuned_v3/figures/*.png
"""

import os
import gc
import csv
import json
import random
from copy import deepcopy
from dataclasses import dataclass, asdict

import numpy as np
import torch
from PIL import Image, ImageDraw

# ============================================================
# CONFIG
# ============================================================
VAL_JSON = "val_prompts_5.json"
OUT_DIR = "outputs_compare_factorized_vs_hybrid_regret_tuned_v3"
FIG_DIR = os.path.join(OUT_DIR, "figures")
CSV_PATH = os.path.join(OUT_DIR, "compare_metrics.csv")

MODEL_NAME = "sd-v1.5-4view"
DEVICE = "cuda:0"
SEED = 23
NUM_FRAMES = 4
SIZE = 256

# payload assumptions
Z_SHARED_BYTES = 946176
Z_VIEW_BYTES = 256
LOCAL_TRANSFORM_BYTES = 0

# 10-step everywhere
DDIM_STEPS_VIEW = 10
DDIM_STEPS_REFRESH = 10
DDIM_STEPS_REFERENCE = 10
GUIDANCE_SCALE = 10.0

# rollout
N_STEPS = 8
FAILURE_STEP = 5

# ============================================================
# LOSS-ALIGNED AIF SETTINGS
# These should match the adaptive regret loss philosophy
# ============================================================
AR_W_REF = 1.00
AR_W_BITS = 0.35
AR_W_LAT = 0.25
AR_W_TEMP = 0.20

# fixed online normalization scales
REF_ERR_SCALE = 0.05
DRIFT_ERR_SCALE = 0.03
LAT_MS_SCALE = 4000.0
BITS_SCALE = float(Z_SHARED_BYTES * 8)

# conservative local-transform gate
LOCAL_ONLY_IF_MOTION_LT = 0.012
LOCAL_ONLY_IF_REF_LT = 0.008
LOCAL_ONLY_IF_DRIFT_LT = 0.006
MAX_LOCAL_STREAK = 0

SCENE_CONF_REFRESH_TH = 0.72
REF_ERR_REFRESH_TH = 0.020
DRIFT_REFRESH_TH = 0.015

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bytes_to_bits(n_bytes: int) -> int:
    return int(n_bytes * 8)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def l2_image_diff(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))

def mean_frame_diff(frames_a, frames_b) -> float:
    vals = []
    for fa, fb in zip(frames_a, frames_b):
        vals.append(l2_image_diff(fa, fb))
    return float(np.mean(vals)) if vals else 0.0

def save_views(out_dir: str, item_id: str, method: str, step_idx: int, frames) -> None:
    d = os.path.join(out_dir, item_id, method)
    ensure_dir(d)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(
            os.path.join(d, f"{item_id}_{method}_step_{step_idx:02d}_view_{i:02d}.png")
        )

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _norm_ref_err(x: float) -> float:
    return _clip01(x / REF_ERR_SCALE)

def _norm_drift_err(x: float) -> float:
    return _clip01(x / DRIFT_ERR_SCALE)

def _norm_lat_ms(x: float) -> float:
    return _clip01(x / LAT_MS_SCALE)

def _norm_bits(x_bits: int) -> float:
    return _clip01(float(x_bits) / BITS_SCALE)

# ============================================================
# STATE
# ============================================================
@dataclass
class SceneState:
    prompt: str
    frames: list
    camera: torch.Tensor | None
    z_shared: object | None = None  # placeholder for future explicit latent reuse

@dataclass
class BeliefState:
    scene_confidence: float = 0.95
    transform_confidence: float = 0.92
    network_quality: float = 0.85
    shared_sent: bool = False
    cumulative_bits: int = 0
    last_action: str = "none"
    recent_ref_error: float = 0.0
    recent_drift_error: float = 0.0
    local_streak: int = 0

@dataclass
class StepRow:
    obj_id: str
    method: str
    step: int
    action: str
    payload_bits: int
    cumulative_bits: int
    motion_mag: float
    proc_ms_total: float
    sample_ms_total: float
    decode_ms_total: float
    local_ms_total: float
    recon_error_vs_prev: float
    ref_error: float

@dataclass
class CandidateResult:
    action: str
    frames: list
    scene_state: SceneState
    payload_bits: int
    proc_ms_total: float
    sample_ms_total: float
    decode_ms_total: float
    local_ms_total: float
    ref_error: float
    drift_error: float
    score: float

# ============================================================
# MODEL INIT
# ============================================================
_STATE = {"ready": False}

def init_model():
    global _STATE
    if _STATE["ready"]:
        return

    from mvdream.camera_utils import get_camera
    from mvdream.ldm.models.diffusion.ddim import DDIMSampler
    from mvdream.model_zoo import build_model

    device = torch.device(DEVICE)
    model = build_model(MODEL_NAME, ckpt_path=None)
    model.factorize_eps = True
    model = model.to(device)
    model.device = device
    model.cond_stage_model = model.cond_stage_model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    uc = model.cond_stage_model([""])
    camera0 = get_camera(NUM_FRAMES, elevation=15, azimuth_start=90, azimuth_span=360).to(device)

    _STATE.update({
        "ready": True,
        "device": device,
        "model": model,
        "sampler": sampler,
        "uc": uc,
        "camera0": camera0,
    })

def get_camera_for_step(step_idx: int):
    init_model()
    base = _STATE["camera0"].clone()
    if base.ndim == 2 and base.shape[1] > 0:
        delta = 2.5 * step_idx
        base[:, -1] = base[:, -1] + delta
    return base

def camera_motion_mag(cam_prev: torch.Tensor, cam_cur: torch.Tensor) -> float:
    if cam_prev is None or cam_cur is None:
        return 1.0
    if cam_prev.shape != cam_cur.shape:
        return 1.0
    diff = torch.norm(cam_cur - cam_prev, dim=-1).mean().item()
    base = torch.norm(cam_prev, dim=-1).mean().item() + 1e-8
    return float(diff / base)

# ============================================================
# RENDERER
# ============================================================
def render_mvdream_views(prompt: str, camera: torch.Tensor, ddim_steps: int):
    init_model()
    st = _STATE
    model = st["model"]
    sampler = st["sampler"]
    uc = st["uc"]

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
            S=int(ddim_steps),
            conditioning=c_,
            batch_size=eff_bs,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=GUIDANCE_SCALE,
            unconditional_conditioning=uc_,
            eta=0.0,
            x_T=None,
        )
        ev1.record()
        torch.cuda.synchronize()
        sample_ms = float(ev0.elapsed_time(ev1))

        ev2 = torch.cuda.Event(enable_timing=True)
        ev3 = torch.cuda.Event(enable_timing=True)
        ev2.record()
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, 0.0, 1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()
        x_sample = x_sample.astype(np.uint8)
        ev3.record()
        torch.cuda.synchronize()
        decode_ms = float(ev2.elapsed_time(ev3))

        frames = [x_sample[i] for i in range(min(NUM_FRAMES, x_sample.shape[0]))]
        return frames, sample_ms, decode_ms

# ============================================================
# STATEFUL SCENE HELPERS
# ============================================================
def initialize_scene(prompt: str, camera: torch.Tensor, ddim_steps: int = DDIM_STEPS_REFERENCE):
    frames, sample_ms, decode_ms = render_mvdream_views(prompt, camera, ddim_steps)
    state = SceneState(prompt=prompt, frames=frames, camera=camera.clone(), z_shared=None)
    return state, sample_ms, decode_ms

def clone_scene_state(state: SceneState) -> SceneState:
    return deepcopy(state)

def render_from_state(state: SceneState, camera: torch.Tensor, ddim_steps: int):
    # Placeholder for future explicit shared-latent reuse
    frames, sample_ms, decode_ms = render_mvdream_views(state.prompt, camera, ddim_steps)
    state.frames = frames
    state.camera = camera.clone()
    return frames, sample_ms, decode_ms

# ============================================================
# LOCAL TRANSFORM
# ============================================================
def affine_warp_numpy(img: np.ndarray, shift_x: int, shift_y: int):
    h, w, c = img.shape
    out = np.full_like(img, 245)

    xs0 = max(0, -shift_x)
    xs1 = min(w, w - shift_x) if shift_x >= 0 else w
    xd0 = max(0, shift_x)
    xd1 = min(w, w + shift_x) if shift_x < 0 else w

    ys0 = max(0, -shift_y)
    ys1 = min(h, h - shift_y) if shift_y >= 0 else h
    yd0 = max(0, shift_y)
    yd1 = min(h, h + shift_y) if shift_y < 0 else h

    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out

def subtle_local_transform(prev_frames, cam_prev, cam_cur):
    motion = camera_motion_mag(cam_prev, cam_cur)
    dx = int(round(8.0 * motion))
    dy = int(round(3.0 * motion))
    out = []
    for i, fr in enumerate(prev_frames):
        sx = dx if i % 2 == 0 else -dx
        sy = dy if i < 2 else -dy
        warped = affine_warp_numpy(fr, sx, sy)
        alpha = 1.0 + min(0.03, 0.08 * motion)
        warped = np.clip(alpha * warped.astype(np.float32), 0, 255).astype(np.uint8)
        out.append(warped)
    return out

# ============================================================
# STEP LOSS SCORING
# ============================================================
def score_step_loss(
    ref_error: float,
    drift_error: float,
    payload_bits: int,
    proc_ms_total: float,
    beliefs: BeliefState,
    action: str,
) -> float:
    ref_term = AR_W_REF * _norm_ref_err(ref_error)
    drift_term = AR_W_TEMP * _norm_drift_err(drift_error)
    bits_term = AR_W_BITS * _norm_bits(payload_bits)
    lat_term = AR_W_LAT * _norm_lat_ms(proc_ms_total)

    switch_pen = 0.0 if beliefs.last_action in ("none", action) else 0.015

    refresh_bonus = 0.0
    if action == "refresh_shared":
        refresh_bonus = 0.06 * max(0.0, 0.85 - beliefs.scene_confidence)

    return float(ref_term + drift_term + bits_term + lat_term + switch_pen - refresh_bonus)

def update_beliefs(
    beliefs: BeliefState,
    motion_mag: float,
    drift_error: float,
    ref_error: float,
    action: str,
):
    beliefs.recent_ref_error = 0.75 * beliefs.recent_ref_error + 0.25 * float(ref_error)
    beliefs.recent_drift_error = 0.75 * beliefs.recent_drift_error + 0.25 * float(drift_error)

    beliefs.scene_confidence = float(
        np.clip(
            0.85 * beliefs.scene_confidence + 0.15 * (1.0 - min(1.0, 12.0 * ref_error)),
            0.05,
            0.99,
        )
    )
    beliefs.transform_confidence = float(
        np.clip(
            0.85 * beliefs.transform_confidence + 0.15 * (1.0 - min(1.0, 10.0 * drift_error)),
            0.05,
            0.99,
        )
    )

    if action == "local_transform":
        beliefs.local_streak += 1
    else:
        beliefs.local_streak = 0

    if action in ("send_view", "refresh_shared"):
        beliefs.shared_sent = True

    if action == "refresh_shared":
        beliefs.scene_confidence = min(0.995, beliefs.scene_confidence + 0.08)
        beliefs.transform_confidence = min(0.995, beliefs.transform_confidence + 0.06)
        beliefs.recent_ref_error *= 0.5
        beliefs.recent_drift_error *= 0.5

    beliefs.last_action = action
    return beliefs

# ============================================================
# CANDIDATE EVALUATION
# ============================================================
def evaluate_local_candidate(
    scene_state,
    beliefs: BeliefState,
    prev_frames,
    cam_prev,
    cam_cur,
    ref_frames,
):
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    frames = subtle_local_transform(prev_frames, cam_prev, cam_cur)
    t1.record()
    torch.cuda.synchronize()
    local_ms = float(t0.elapsed_time(t1))

    drift_error = 0.0 if prev_frames is None else mean_frame_diff(prev_frames, frames)
    ref_error = mean_frame_diff(frames, ref_frames)
    payload_bits = int(LOCAL_TRANSFORM_BYTES * 8)
    proc_ms_total = float(local_ms)

    st2 = clone_scene_state(scene_state)
    st2.frames = [fr.copy() for fr in frames]
    st2.camera = cam_cur.clone()

    score = score_step_loss(
        ref_error=ref_error,
        drift_error=drift_error,
        payload_bits=payload_bits,
        proc_ms_total=proc_ms_total,
        beliefs=beliefs,
        action="local_transform",
    )

    return CandidateResult(
        action="local_transform",
        frames=frames,
        scene_state=st2,
        payload_bits=payload_bits,
        proc_ms_total=proc_ms_total,
        sample_ms_total=0.0,
        decode_ms_total=0.0,
        local_ms_total=local_ms,
        ref_error=ref_error,
        drift_error=drift_error,
        score=score,
    )

def evaluate_render_candidate(
    action: str,
    scene_state,
    beliefs: BeliefState,
    prev_frames,
    cam_cur,
    ref_frames,
):
    assert action in ("send_view", "refresh_shared")

    st2 = clone_scene_state(scene_state)
    frames, sample_ms, decode_ms = render_from_state(
        st2,
        cam_cur,
        ddim_steps=DDIM_STEPS_VIEW if action == "send_view" else DDIM_STEPS_REFRESH,
    )

    drift_error = 0.0 if prev_frames is None else mean_frame_diff(prev_frames, frames)
    ref_error = mean_frame_diff(frames, ref_frames)

    payload_bits = int((Z_VIEW_BYTES if action == "send_view" else Z_SHARED_BYTES) * 8)
    proc_ms_total = float(sample_ms + decode_ms)

    score = score_step_loss(
        ref_error=ref_error,
        drift_error=drift_error,
        payload_bits=payload_bits,
        proc_ms_total=proc_ms_total,
        beliefs=beliefs,
        action=action,
    )

    return CandidateResult(
        action=action,
        frames=frames,
        scene_state=st2,
        payload_bits=payload_bits,
        proc_ms_total=proc_ms_total,
        sample_ms_total=sample_ms,
        decode_ms_total=decode_ms,
        local_ms_total=0.0,
        ref_error=ref_error,
        drift_error=drift_error,
        score=score,
    )

def choose_hybrid_candidate(
    scene_state,
    beliefs: BeliefState,
    prev_frames,
    cam_prev,
    cam_cur,
    ref_frames,
):
    motion_mag = camera_motion_mag(cam_prev, cam_cur)
    candidates = []

    candidates.append(
        evaluate_render_candidate(
            action="send_view",
            scene_state=scene_state,
            beliefs=beliefs,
            prev_frames=prev_frames,
            cam_cur=cam_cur,
            ref_frames=ref_frames,
        )
    )

    need_refresh = (
        (not beliefs.shared_sent)
        or (beliefs.scene_confidence < SCENE_CONF_REFRESH_TH)
        or (beliefs.recent_ref_error > REF_ERR_REFRESH_TH)
        or (beliefs.recent_drift_error > DRIFT_REFRESH_TH)
    )
    if need_refresh:
        candidates.append(
            evaluate_render_candidate(
                action="refresh_shared",
                scene_state=scene_state,
                beliefs=beliefs,
                prev_frames=prev_frames,
                cam_cur=cam_cur,
                ref_frames=ref_frames,
            )
        )

    local_allowed = (
        prev_frames is not None
        and motion_mag < LOCAL_ONLY_IF_MOTION_LT
        and beliefs.recent_ref_error < LOCAL_ONLY_IF_REF_LT
        and beliefs.recent_drift_error < LOCAL_ONLY_IF_DRIFT_LT
        and beliefs.local_streak <= MAX_LOCAL_STREAK
    )
    if local_allowed:
        candidates.append(
            evaluate_local_candidate(
                scene_state=scene_state,
                beliefs=beliefs,
                prev_frames=prev_frames,
                cam_prev=cam_prev,
                cam_cur=cam_cur,
                ref_frames=ref_frames,
            )
        )

    best = min(candidates, key=lambda c: c.score)
    return best

# ============================================================
# POLICIES
# ============================================================
def choose_factorized_only_action(step_idx: int) -> str:
    if step_idx == 0:
        return "refresh_shared"
    return "send_view"

# ============================================================
# REFERENCE
# ============================================================
def get_reference_frames(scene_state: SceneState, camera: torch.Tensor):
    frames, _, _ = render_from_state(scene_state, camera, ddim_steps=DDIM_STEPS_REFERENCE)
    return [fr.copy() for fr in frames]

# ============================================================
# STEP EXECUTION
# ============================================================
def run_step(
    method: str,
    item_id: str,
    scene_state,
    step_idx: int,
    beliefs: BeliefState | None,
    prev_frames,
    cam_prev,
    cam_cur,
    ref_frames,
):
    motion_mag = camera_motion_mag(cam_prev, cam_cur)

    if method == "hybrid":
        best = choose_hybrid_candidate(
            scene_state=scene_state,
            beliefs=beliefs,
            prev_frames=prev_frames,
            cam_prev=cam_prev,
            cam_cur=cam_cur,
            ref_frames=ref_frames,
        )

        scene_state.prompt = best.scene_state.prompt
        scene_state.frames = [fr.copy() for fr in best.scene_state.frames]
        scene_state.camera = None if best.scene_state.camera is None else best.scene_state.camera.clone()
        scene_state.z_shared = best.scene_state.z_shared

        frames = best.frames
        action = best.action
        payload_bits = best.payload_bits
        sample_ms = best.sample_ms_total
        decode_ms = best.decode_ms_total
        local_ms = best.local_ms_total
        proc_ms = best.proc_ms_total
        recon_error_vs_prev = best.drift_error
        ref_error = best.ref_error

    elif method == "factorized":
        action = choose_factorized_only_action(step_idx)

        if action == "send_view":
            frames, sample_ms, decode_ms = render_from_state(scene_state, cam_cur, ddim_steps=DDIM_STEPS_VIEW)
            local_ms = 0.0
            payload_bits = int(Z_VIEW_BYTES * 8)
        elif action == "refresh_shared":
            frames, sample_ms, decode_ms = render_from_state(scene_state, cam_cur, ddim_steps=DDIM_STEPS_REFRESH)
            local_ms = 0.0
            payload_bits = int(Z_SHARED_BYTES * 8)
        else:
            raise ValueError(action)

        proc_ms = float(sample_ms + decode_ms)
        recon_error_vs_prev = 0.0 if prev_frames is None else mean_frame_diff(prev_frames, frames)
        ref_error = mean_frame_diff(frames, ref_frames)

    else:
        raise ValueError(method)

    if beliefs is not None:
        beliefs.cumulative_bits += int(payload_bits)
        beliefs = update_beliefs(
            beliefs=beliefs,
            motion_mag=motion_mag,
            drift_error=recon_error_vs_prev,
            ref_error=ref_error,
            action=action,
        )
        cumulative_bits = beliefs.cumulative_bits
    else:
        cumulative_bits = 0

    row = StepRow(
        obj_id=item_id,
        method=method,
        step=step_idx,
        action=action,
        payload_bits=int(payload_bits),
        cumulative_bits=int(cumulative_bits),
        motion_mag=float(motion_mag),
        proc_ms_total=float(proc_ms),
        sample_ms_total=float(sample_ms),
        decode_ms_total=float(decode_ms),
        local_ms_total=float(local_ms),
        recon_error_vs_prev=float(recon_error_vs_prev),
        ref_error=float(ref_error),
    )

    return frames, beliefs, row

# ============================================================
# FIGURE HELPERS
# ============================================================
def add_label_band(img: Image.Image, text: str, band_h: int = 28) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + band_h), (255, 255, 255))
    out.paste(img, (0, band_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), text, fill=(0, 0, 0))
    return out

def make_contact_strip(frames, title: str) -> Image.Image:
    ims = [Image.fromarray(fr) for fr in frames]
    w, h = ims[0].size
    strip = Image.new("RGB", (w * len(ims), h), (255, 255, 255))
    for i, im in enumerate(ims):
        strip.paste(im, (i * w, 0))
    strip = add_label_band(strip, title, band_h=32)
    return strip

def make_failure_case_figure(item_id: str, step_idx: int, ref_frames, fac_frames, hyb_frames,
                             fac_action: str, hyb_action: str,
                             fac_ref_err: float, hyb_ref_err: float):
    row1 = make_contact_strip(ref_frames, f"Reference (step {step_idx})")
    row2 = make_contact_strip(
        fac_frames,
        f"Factorized-only | action={fac_action} | ref_err={fac_ref_err:.4f}"
    )
    row3 = make_contact_strip(
        hyb_frames,
        f"Hybrid AIF loss-aligned | action={hyb_action} | ref_err={hyb_ref_err:.4f}"
    )
    W = max(row1.width, row2.width, row3.width)
    H = row1.height + row2.height + row3.height + 20
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for row in [row1, row2, row3]:
        canvas.paste(row, (0, y))
        y += row.height + 10
    out_path = os.path.join(FIG_DIR, f"{item_id}_failure_step_{step_idx:02d}.png")
    canvas.save(out_path)
    return out_path

# ============================================================
# MAIN COMPARISON
# ============================================================
def compare_one_item(item_id: str, prompt: str):
    if "3d asset" not in prompt.lower():
        prompt = prompt.rstrip(".") + ", 3d asset"

    factorized_beliefs = BeliefState(shared_sent=False)
    hybrid_beliefs = BeliefState(shared_sent=False)

    init_cam = get_camera_for_step(0)
    base_scene, _, _ = initialize_scene(prompt, init_cam, ddim_steps=DDIM_STEPS_REFERENCE)

    ref_scene = clone_scene_state(base_scene)
    fac_scene = clone_scene_state(base_scene)
    hyb_scene = clone_scene_state(base_scene)

    fac_prev = None
    hyb_prev = None
    cam_prev = None
    rows = []
    saved = {}

    for step_idx in range(N_STEPS):
        cam_cur = get_camera_for_step(step_idx)

        ref_frames = get_reference_frames(ref_scene, cam_cur)

        fac_frames, factorized_beliefs, fac_row = run_step(
            method="factorized",
            item_id=item_id,
            scene_state=fac_scene,
            step_idx=step_idx,
            beliefs=factorized_beliefs,
            prev_frames=fac_prev,
            cam_prev=cam_prev,
            cam_cur=cam_cur,
            ref_frames=ref_frames,
        )

        hyb_frames, hybrid_beliefs, hyb_row = run_step(
            method="hybrid",
            item_id=item_id,
            scene_state=hyb_scene,
            step_idx=step_idx,
            beliefs=hybrid_beliefs,
            prev_frames=hyb_prev,
            cam_prev=cam_prev,
            cam_cur=cam_cur,
            ref_frames=ref_frames,
        )

        save_views(OUT_DIR, item_id, "reference", step_idx, ref_frames)
        save_views(OUT_DIR, item_id, "factorized", step_idx, fac_frames)
        save_views(OUT_DIR, item_id, "hybrid", step_idx, hyb_frames)

        rows.append(asdict(fac_row))
        rows.append(asdict(hyb_row))

        if step_idx == FAILURE_STEP:
            saved = {
                "ref_frames": ref_frames,
                "fac_frames": fac_frames,
                "hyb_frames": hyb_frames,
                "fac_action": fac_row.action,
                "hyb_action": hyb_row.action,
                "fac_ref_err": fac_row.ref_error,
                "hyb_ref_err": hyb_row.ref_error,
            }

        fac_prev = [fr.copy() for fr in fac_frames]
        hyb_prev = [fr.copy() for fr in hyb_frames]
        cam_prev = cam_cur.clone()

        gc.collect()
        torch.cuda.empty_cache()

    fig_path = None
    if saved:
        fig_path = make_failure_case_figure(
            item_id=item_id,
            step_idx=FAILURE_STEP,
            ref_frames=saved["ref_frames"],
            fac_frames=saved["fac_frames"],
            hyb_frames=saved["hyb_frames"],
            fac_action=saved["fac_action"],
            hyb_action=saved["hyb_action"],
            fac_ref_err=saved["fac_ref_err"],
            hyb_ref_err=saved["hyb_ref_err"],
        )

    return rows, fig_path

def main():
    set_seed(SEED)
    init_model()

    with open(VAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = []
    figure_paths = []

    for item in data[:5]:
        item_id = item["id"]
        prompt = item["prompt"].strip()
        print(f"Comparing item: {item_id}")
        rows, fig_path = compare_one_item(item_id, prompt)
        all_rows.extend(rows)
        if fig_path is not None:
            figure_paths.append(fig_path)

    if all_rows:
        with open(CSV_PATH, "w", newline="") as f:
            fieldnames = list(all_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Saved metrics to: {CSV_PATH}")

    for p in figure_paths:
        print(f"Saved figure: {p}")

if __name__ == "__main__":
    main()

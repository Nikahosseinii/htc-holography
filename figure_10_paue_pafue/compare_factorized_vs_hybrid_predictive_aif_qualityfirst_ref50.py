import os
import gc
import csv
import json
import random
import hashlib
from dataclasses import dataclass, asdict

import numpy as np
import torch
from PIL import Image, ImageDraw

# ============================================================
# CONFIG
# ============================================================
VAL_JSON = "val_prompts_5.json"

# New output folder so we do NOT overwrite the previous zero-error version
OUT_DIR = "outputs_compare_factorized_vs_hybrid_predictive_aif_qualityfirst_ref50"
FIG_DIR = os.path.join(OUT_DIR, "figures")
CSV_PATH = os.path.join(OUT_DIR, "compare_metrics.csv")

MODEL_NAME = "sd-v1.5-4view"
DEVICE = "cuda:0"
SEED = 23

NUM_FRAMES = 4
SIZE = 256

# Payload assumptions
Z_SHARED_BYTES = 946176
Z_VIEW_BYTES = 256
LOCAL_TRANSFORM_BYTES = 0

# Method settings
DDIM_STEPS_VIEW = 10          # fixed factorized view update
DDIM_STEPS_REFRESH = 25       # Hybrid AIF full shared refresh
DDIM_STEPS_REF = 50           # high-fidelity reference, avoids artificial zero error
GUIDANCE_SCALE = 10.0

# Comparison settings
N_STEPS = 8
FAILURE_STEP = 5

# ============================================================
# QUALITY-FIRST PREDICTIVE AIF SETTINGS
# ============================================================
# This policy is tuned for PAUE/PAFUE, but the reference is now 50-step,
# while Hybrid AIF can only use 10-step view update or 25-step refresh.
# Therefore Hybrid cannot exactly equal the reference.

W_RISK = 4.2
W_AMBIG = 1.4
W_BW = 0.10
W_LAT = 0.08

SMALL_MOTION_TH = 0.020
LOW_CONF_TH = 0.78

MAX_CONSEC_LOCAL = 0
MAX_STEPS_WITHOUT_REFRESH = 1

LOCAL_ERROR_TH = 0.020
REFRESH_ERROR_TH = 0.035

HIGH_PRED_UNCERTAINTY_TH = 0.35
LOCAL_PRED_UNCERTAINTY_TH = 0.10

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================
def stable_seed(*parts) -> int:
    text = "::".join(str(p) for p in parts)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
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


# ============================================================
# BELIEF STATE
# ============================================================
@dataclass
class BeliefState:
    scene_confidence: float = 0.95
    transform_confidence: float = 0.90
    network_quality: float = 0.80
    shared_sent: bool = False
    cumulative_bits: int = 0
    last_action: str = "none"

    consecutive_local: int = 0
    steps_since_refresh: int = 999
    last_recon_error: float = 0.0
    ema_recon_error: float = 0.0
    ema_motion: float = 0.0

    pred_uncertainty: float = 0.0
    ema_pred_uncertainty: float = 0.0
    pred_change_mag: float = 0.0


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
    pred_uncertainty: float = 0.0
    ema_pred_uncertainty: float = 0.0
    pred_change_mag: float = 0.0


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

    camera0 = get_camera(
        NUM_FRAMES,
        elevation=15,
        azimuth_start=90,
        azimuth_span=360,
    ).to(device)

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
# MVDREAM RENDERER
# ============================================================
def render_mvdream_views(prompt: str, camera: torch.Tensor, ddim_steps: int, render_seed: int):
    """
    Deterministic rendering for fair comparison.

    The same seed is used for the same object and step, but the methods use
    different DDIM step counts:
    - reference: 50 steps
    - Hybrid refresh: 25 steps
    - factorized view update: 10 steps

    This avoids the artificial zero-error case.
    """
    init_model()
    set_seed(render_seed)

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

        torch.cuda.synchronize()
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

    dx = int(round(6.0 * motion))
    dy = int(round(2.0 * motion))

    out = []

    for i, fr in enumerate(prev_frames):
        sx = dx if i % 2 == 0 else -dx
        sy = dy if i < 2 else -dy

        warped = affine_warp_numpy(fr, sx, sy)

        alpha = 1.0 + min(0.015, 0.04 * motion)
        warped = np.clip(alpha * warped.astype(np.float32), 0, 255).astype(np.uint8)

        out.append(warped)

    return out


# ============================================================
# MOTION-COMPENSATED PREDICTION
# ============================================================
def predict_next_frames(prev_frames, cam_prev, cam_cur):
    if prev_frames is None or cam_prev is None or cam_cur is None:
        return None
    return subtle_local_transform(prev_frames, cam_prev, cam_cur)


def estimate_prediction_uncertainty(prev_frames, pred_frames, motion_mag: float):
    if prev_frames is None or pred_frames is None:
        return 1.0, 1.0

    pred_change = mean_frame_diff(prev_frames, pred_frames)

    change_term = min(1.0, pred_change * 50.0)
    motion_term = min(1.0, motion_mag * 8.0)

    pred_uncertainty = float(
        np.clip(
            0.70 * change_term + 0.30 * motion_term,
            0.0,
            1.0,
        )
    )

    return pred_uncertainty, float(pred_change)


def update_prediction_belief(
    beliefs: BeliefState,
    prev_frames,
    cam_prev,
    cam_cur,
    motion_mag: float,
):
    pred_frames = predict_next_frames(prev_frames, cam_prev, cam_cur)

    pred_uncertainty, pred_change = estimate_prediction_uncertainty(
        prev_frames=prev_frames,
        pred_frames=pred_frames,
        motion_mag=motion_mag,
    )

    beliefs.pred_uncertainty = float(pred_uncertainty)
    beliefs.pred_change_mag = float(pred_change)

    beliefs.ema_pred_uncertainty = float(
        np.clip(
            0.75 * beliefs.ema_pred_uncertainty + 0.25 * pred_uncertainty,
            0.0,
            1.0,
        )
    )

    return beliefs, pred_frames


# ============================================================
# QUALITY-FIRST PREDICTIVE AIF POLICY
# ============================================================
def update_beliefs(
    beliefs: BeliefState,
    motion_mag: float,
    recon_error: float | None,
    action: str,
):
    beliefs.ema_motion = float(
        np.clip(0.75 * beliefs.ema_motion + 0.25 * motion_mag, 0.0, 1.0)
    )

    if recon_error is not None:
        beliefs.last_recon_error = float(recon_error)
        beliefs.ema_recon_error = float(
            0.75 * beliefs.ema_recon_error + 0.25 * recon_error
        )

    err_term = min(1.0, beliefs.ema_recon_error * 25.0)
    motion_term = min(1.0, beliefs.ema_motion * 6.0)
    pred_term = min(1.0, beliefs.ema_pred_uncertainty)

    beliefs.transform_confidence = float(
        np.clip(
            0.85 * beliefs.transform_confidence
            + 0.15 * (1.0 - 0.35 * motion_term - 0.35 * err_term - 0.30 * pred_term),
            0.05,
            0.99,
        )
    )

    if recon_error is not None:
        beliefs.scene_confidence = float(
            np.clip(
                0.88 * beliefs.scene_confidence
                + 0.12 * (1.0 - min(1.0, recon_error * 18.0)),
                0.05,
                0.99,
            )
        )

    if action == "local_transform":
        beliefs.consecutive_local += 1
        beliefs.steps_since_refresh += 1
        beliefs.scene_confidence = max(0.05, beliefs.scene_confidence - 0.050)
        beliefs.transform_confidence = max(0.05, beliefs.transform_confidence - 0.060)

    elif action == "send_view":
        beliefs.consecutive_local = 0
        beliefs.steps_since_refresh += 1
        beliefs.scene_confidence = min(0.98, beliefs.scene_confidence + 0.030)
        beliefs.transform_confidence = min(0.98, beliefs.transform_confidence + 0.050)

    elif action == "refresh_shared":
        beliefs.consecutive_local = 0
        beliefs.steps_since_refresh = 0
        beliefs.scene_confidence = min(0.99, beliefs.scene_confidence + 0.30)
        beliefs.transform_confidence = min(0.99, beliefs.transform_confidence + 0.15)
        beliefs.shared_sent = True

    else:
        raise ValueError(action)

    beliefs.last_action = action
    return beliefs


def expected_free_energy(action: str, beliefs: BeliefState, motion_mag: float):
    pred_now = beliefs.pred_uncertainty
    pred_ema = beliefs.ema_pred_uncertainty

    if action == "local_transform":
        payload_bytes = LOCAL_TRANSFORM_BYTES
        lat_cost = 0.03
        risk = (
            0.35
            + 4.00 * motion_mag
            + 1.20 * pred_now
            + 1.00 * pred_ema
            + 1.00 * (1.0 - beliefs.transform_confidence)
            + 0.80 * min(1.0, beliefs.ema_recon_error * 25.0)
            + 0.50 * beliefs.consecutive_local
        )
        ambiguity = (
            1.0
            - 0.45 * beliefs.scene_confidence
            - 0.25 * beliefs.transform_confidence
            + 0.50 * pred_ema
        )

    elif action == "send_view":
        payload_bytes = Z_VIEW_BYTES
        lat_cost = DDIM_STEPS_VIEW / float(DDIM_STEPS_REFRESH)
        risk = (
            0.080
            + 0.35 * motion_mag
            + 0.30 * pred_ema
            + 0.20 * min(1.0, beliefs.ema_recon_error * 20.0)
        )
        ambiguity = (
            0.45 * (1.0 - beliefs.transform_confidence)
            + 0.25 * pred_ema
        )

    elif action == "refresh_shared":
        payload_bytes = Z_SHARED_BYTES
        lat_cost = 1.0
        risk = 0.005
        ambiguity = 0.20 * (1.0 - beliefs.scene_confidence)

    else:
        raise ValueError(action)

    bw_cost = payload_bytes / float(Z_SHARED_BYTES)

    G = (
        W_RISK * risk
        + W_AMBIG * ambiguity
        + W_BW * bw_cost * (1.2 - beliefs.network_quality)
        + W_LAT * lat_cost
    )

    return float(G), int(payload_bytes)


def choose_hybrid_action(beliefs: BeliefState, motion_mag: float) -> str:
    if not beliefs.shared_sent:
        return "refresh_shared"

    if beliefs.scene_confidence < LOW_CONF_TH:
        return "refresh_shared"

    if beliefs.ema_recon_error > REFRESH_ERROR_TH:
        return "refresh_shared"

    if beliefs.ema_pred_uncertainty > HIGH_PRED_UNCERTAINTY_TH:
        return "refresh_shared"

    # Quality-first: refresh frequently, but reference is now 50-step,
    # so Hybrid refresh no longer equals the reference.
    if beliefs.steps_since_refresh >= MAX_STEPS_WITHOUT_REFRESH:
        return "refresh_shared"

    # Local transform is almost never used in the quality-first PAUE setting.
    if (
        motion_mag < SMALL_MOTION_TH
        and beliefs.pred_uncertainty < LOCAL_PRED_UNCERTAINTY_TH
        and beliefs.transform_confidence > 0.90
        and MAX_CONSEC_LOCAL > 0
    ):
        g_local, _ = expected_free_energy("local_transform", beliefs, motion_mag)
        g_view, _ = expected_free_energy("send_view", beliefs, motion_mag)
        if g_local + 0.10 < g_view:
            return "local_transform"

    vals = {
        "send_view": expected_free_energy("send_view", beliefs, motion_mag)[0],
        "refresh_shared": expected_free_energy("refresh_shared", beliefs, motion_mag)[0],
    }

    return min(vals, key=vals.get)


# ============================================================
# FACTORIZED-ONLY POLICY
# ============================================================
def choose_factorized_only_action(step_idx: int) -> str:
    if step_idx == 0:
        return "refresh_shared"
    return "send_view"


# ============================================================
# ONE STEP EXECUTION
# ============================================================
def run_step(
    method: str,
    item_id: str,
    prompt: str,
    step_idx: int,
    beliefs: BeliefState | None,
    prev_frames,
    cam_prev,
    cam_cur,
    ref_frames,
    render_seed: int,
):
    motion_mag = camera_motion_mag(cam_prev, cam_cur)
    pred_frames = None

    if method == "hybrid":
        beliefs, pred_frames = update_prediction_belief(
            beliefs=beliefs,
            prev_frames=prev_frames,
            cam_prev=cam_prev,
            cam_cur=cam_cur,
            motion_mag=motion_mag,
        )
        action = choose_hybrid_action(beliefs, motion_mag)

    elif method == "factorized":
        action = choose_factorized_only_action(step_idx)

    else:
        raise ValueError(method)

    payload_bytes = 0
    sample_ms = 0.0
    decode_ms = 0.0
    local_ms = 0.0
    frames = prev_frames

    if action == "local_transform":
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)

        t0.record()
        frames = pred_frames if pred_frames is not None else subtle_local_transform(prev_frames, cam_prev, cam_cur)
        t1.record()

        torch.cuda.synchronize()
        local_ms = float(t0.elapsed_time(t1))
        payload_bytes = LOCAL_TRANSFORM_BYTES

    elif action == "send_view":
        frames, sample_ms, decode_ms = render_mvdream_views(
            prompt=prompt,
            camera=cam_cur,
            ddim_steps=DDIM_STEPS_VIEW,
            render_seed=render_seed,
        )
        payload_bytes = Z_VIEW_BYTES

    elif action == "refresh_shared":
        frames, sample_ms, decode_ms = render_mvdream_views(
            prompt=prompt,
            camera=cam_cur,
            ddim_steps=DDIM_STEPS_REFRESH,
            render_seed=render_seed,
        )
        payload_bytes = Z_SHARED_BYTES

    else:
        raise ValueError(action)

    recon_error_vs_prev = 0.0
    if prev_frames is not None and frames is not None:
        recon_error_vs_prev = mean_frame_diff(prev_frames, frames)

    ref_error = mean_frame_diff(frames, ref_frames) if ref_frames is not None else 0.0

    proc_ms = float(sample_ms + decode_ms + local_ms)

    if beliefs is not None:
        beliefs.cumulative_bits += bytes_to_bits(payload_bytes)
        beliefs = update_beliefs(
            beliefs=beliefs,
            motion_mag=motion_mag,
            recon_error=recon_error_vs_prev,
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
        payload_bits=bytes_to_bits(payload_bytes),
        cumulative_bits=cumulative_bits,
        motion_mag=motion_mag,
        proc_ms_total=proc_ms,
        sample_ms_total=sample_ms,
        decode_ms_total=decode_ms,
        local_ms_total=local_ms,
        recon_error_vs_prev=float(recon_error_vs_prev),
        ref_error=float(ref_error),
        pred_uncertainty=float(beliefs.pred_uncertainty) if beliefs is not None else 0.0,
        ema_pred_uncertainty=float(beliefs.ema_pred_uncertainty) if beliefs is not None else 0.0,
        pred_change_mag=float(beliefs.pred_change_mag) if beliefs is not None else 0.0,
    )

    return frames, beliefs, row


# ============================================================
# REFERENCE STRATEGY
# ============================================================
def get_reference_frames(prompt: str, camera: torch.Tensor, render_seed: int):
    """
    Fair high-fidelity reference.

    Reference uses 50 DDIM steps.
    Hybrid AIF refresh uses 25 DDIM steps.
    Fixed factorized view update uses 10 DDIM steps.

    Therefore Hybrid AIF cannot exactly equal the reference.
    """
    frames, _, _ = render_mvdream_views(
        prompt=prompt,
        camera=camera,
        ddim_steps=DDIM_STEPS_REF,
        render_seed=render_seed,
    )
    return frames


# ============================================================
# FIGURE MAKER
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


def make_failure_case_figure(
    item_id: str,
    step_idx: int,
    ref_frames,
    fac_frames,
    hyb_frames,
    fac_action: str,
    hyb_action: str,
    fac_ref_err: float,
    hyb_ref_err: float,
):
    row1 = make_contact_strip(ref_frames, f"Reference 50-step (step {step_idx})")

    row2 = make_contact_strip(
        fac_frames,
        f"Fixed Factorized | action={fac_action} | ref_err={fac_ref_err:.4f}",
    )

    row3 = make_contact_strip(
        hyb_frames,
        f"Quality-first Predictive AIF | action={hyb_action} | ref_err={hyb_ref_err:.4f}",
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

    fac_prev = None
    hyb_prev = None
    cam_prev = None

    rows = []
    saved = {}

    for step_idx in range(N_STEPS):
        cam_cur = get_camera_for_step(step_idx)

        render_seed = stable_seed(SEED, item_id, step_idx, "canonical")

        ref_frames = get_reference_frames(
            prompt=prompt,
            camera=cam_cur,
            render_seed=render_seed,
        )

        fac_frames, factorized_beliefs, fac_row = run_step(
            method="factorized",
            item_id=item_id,
            prompt=prompt,
            step_idx=step_idx,
            beliefs=factorized_beliefs,
            prev_frames=fac_prev,
            cam_prev=cam_prev,
            cam_cur=cam_cur,
            ref_frames=ref_frames,
            render_seed=render_seed,
        )

        hyb_frames, hybrid_beliefs, hyb_row = run_step(
            method="hybrid",
            item_id=item_id,
            prompt=prompt,
            step_idx=step_idx,
            beliefs=hybrid_beliefs,
            prev_frames=hyb_prev,
            cam_prev=cam_prev,
            cam_cur=cam_cur,
            ref_frames=ref_frames,
            render_seed=render_seed,
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

        fac_prev = fac_frames
        hyb_prev = hyb_frames
        cam_prev = cam_cur

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
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"Saved metrics to: {CSV_PATH}")

    for p in figure_paths:
        print(f"Saved figure: {p}")


if __name__ == "__main__":
    main()

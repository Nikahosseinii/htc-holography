import matplotlib.pyplot as plt
import numpy as np

# Bigger readable fonts
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# -----------------------------
# Parameters
# -----------------------------
seconds = 5
num_points = 500
t = np.linspace(0, seconds, num_points)

fps1 = 30
fps2 = 60

# -----------------------------
# Payload sizes (bytes)
# -----------------------------
rgb = 512 * 512 * 3                  # 786432 B per frame
x0 = 4 * (512 // 8) * (512 // 8) * 2 # 32768 B per frame
z_shared = 946176                    # one-time
z_view = 256                         # per update

# -----------------------------
# Helpers
# -----------------------------
def to_mb(x):
    return x / (1024 ** 2)

def cumulative_stream(payload_per_frame, fps, t_axis):
    return payload_per_frame * fps * t_axis

def cumulative_proposed(shared_payload, update_payload, fps, t_axis):
    return shared_payload + update_payload * fps * t_axis

def cumulative_updates_only(update_payload, fps, t_axis):
    return update_payload * fps * t_axis

def break_even_time(stream_payload, stream_fps, shared_payload, update_payload, update_fps):
    denom = stream_payload * stream_fps - update_payload * update_fps
    if denom <= 0:
        return None
    t_be = shared_payload / denom
    return t_be if t_be >= 0 else None

# -----------------------------
# Curves (bytes)
# -----------------------------
cum_rgb_30 = cumulative_stream(rgb, fps1, t)
cum_rgb_60 = cumulative_stream(rgb, fps2, t)
cum_x0_30 = cumulative_stream(x0, fps1, t)
cum_x0_60 = cumulative_stream(x0, fps2, t)

cum_prop_30 = cumulative_proposed(z_shared, z_view, fps1, t)
cum_prop_updates_30 = cumulative_updates_only(z_view, fps1, t)

# Convert to MB
cum_rgb_30_mb = to_mb(cum_rgb_30)
cum_rgb_60_mb = to_mb(cum_rgb_60)
cum_x0_30_mb = to_mb(cum_x0_30)
cum_x0_60_mb = to_mb(cum_x0_60)
cum_prop_30_mb = to_mb(cum_prop_30)
cum_prop_updates_30_mb = to_mb(cum_prop_updates_30)

# Break-even times
t_be_rgb30 = break_even_time(rgb, fps1, z_shared, z_view, fps1)
t_be_x030 = break_even_time(x0, fps1, z_shared, z_view, fps1)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10.5, 6.5))

ax.plot(t, cum_rgb_30_mb, linewidth=2.5, label="RGB streaming (30 FPS)")
ax.plot(t, cum_rgb_60_mb, linewidth=2.5, label="RGB streaming (60 FPS)")
ax.plot(t, cum_x0_30_mb, linewidth=2.5, label="Latent streaming (30 FPS)")
ax.plot(t, cum_x0_60_mb, linewidth=2.5, label="Latent streaming (60 FPS)")
ax.plot(t, cum_prop_30_mb, linewidth=3.0, label="Proposed: one-time + updates (30 FPS)")
ax.plot(t, cum_prop_updates_30_mb, linewidth=2.5, label="Proposed: updates only (30 FPS)")

# Initial shared-latent marker
ax.scatter([0], [to_mb(z_shared)], s=65, zorder=5)

# Break-even markers
selected_break_even = [
    ("RGB 30 FPS", t_be_rgb30, rgb, fps1),
    ("Latent 30 FPS", t_be_x030, x0, fps1),
]

for name, t_be, payload, fps in selected_break_even:
    if t_be is not None and t_be <= seconds:
        y_be = to_mb(payload * fps * t_be)
        ax.scatter([t_be], [y_be], s=70, zorder=6)
        ax.annotate(
            f"{name}\n{t_be:.3f} s",
            xy=(t_be, y_be),
            xytext=(t_be + 0.25, y_be * 1.35),
            fontsize=12,
            arrowprops=dict(arrowstyle="->", lw=1),
        )

# Summary values at 5 s
t_end = seconds
rgb30_end = to_mb(rgb * fps1 * t_end)
rgb60_end = to_mb(rgb * fps2 * t_end)
x030_end = to_mb(x0 * fps1 * t_end)
x060_end = to_mb(x0 * fps2 * t_end)
prop30_end = to_mb(z_shared + z_view * fps1 * t_end)

summary_text = (
    f"At {seconds:.0f} s:\n"
    f"RGB 30 FPS = {rgb30_end:.1f} MB\n"
    f"RGB 60 FPS = {rgb60_end:.1f} MB\n"
    f"Latent 30 FPS = {x030_end:.2f} MB\n"
    f"Latent 60 FPS = {x060_end:.2f} MB\n"
    f"Proposed = {prop30_end:.2f} MB"
)

ax.text(
    0.66, 0.28, summary_text,
    transform=ax.transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
)

# Labels and layout
ax.set_xlabel("Time (seconds)", fontsize=16)
ax.set_ylabel("Cumulative transmitted data (MB)", fontsize=16)
ax.set_title("Cumulative transmitted data vs time under different transmission paradigms", fontsize=18)
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.5)

ax.legend(
    loc="lower left",
    bbox_to_anchor=(0.18, 0.02),
    fontsize=12,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig("cumulative_payload_comparison_richer_bigfont.pdf", bbox_inches="tight")
plt.savefig("cumulative_payload_comparison_richer_bigfont.png", dpi=300, bbox_inches="tight")
print("Saved cumulative_payload_comparison_richer_bigfont.pdf/.png")

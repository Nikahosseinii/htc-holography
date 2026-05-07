import matplotlib.pyplot as plt
import numpy as np

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
rgb = 512 * 512 * 3                    # 786432 B per frame
x0 = 4 * (512 // 8) * (512 // 8) * 2   # 32768 B per frame
z_shared = 946176                      # one-time
z_view = 256                           # per update

# -----------------------------
# Helper
# -----------------------------
def to_MB(x):
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
cum_rgb_30_MB = to_MB(cum_rgb_30)
cum_rgb_60_MB = to_MB(cum_rgb_60)
cum_x0_30_MB = to_MB(cum_x0_30)
cum_x0_60_MB = to_MB(cum_x0_60)
cum_prop_30_MB = to_MB(cum_prop_30)
cum_prop_updates_30_MB = to_MB(cum_prop_updates_30)

# -----------------------------
# Break-even times
# -----------------------------
t_be_rgb30 = break_even_time(rgb, fps1, z_shared, z_view, fps1)
t_be_x030 = break_even_time(x0, fps1, z_shared, z_view, fps1)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6.5))

# Keep colors, but make line styles distinct for grayscale print
plt.plot(
    t, cum_rgb_30_MB,
    color='tab:blue', linestyle='-', linewidth=2.2,
    label='RGB streaming (30 FPS)'
)

plt.plot(
    t, cum_rgb_60_MB,
    color='tab:blue', linestyle='--', linewidth=2.2,
    label='RGB streaming (60 FPS)'
)

plt.plot(
    t, cum_x0_30_MB,
    color='tab:green', linestyle=':', linewidth=2.6,
    label='Latent streaming (30 FPS)'
)

plt.plot(
    t, cum_x0_60_MB,
    color='tab:green', linestyle='-.', linewidth=2.2,
    label='Latent streaming (60 FPS)'
)

plt.plot(
    t, cum_prop_30_MB,
    color='tab:red', linestyle=(0, (8, 3)), linewidth=2.5,
    label='Proposed: one-time + updates (30 FPS)'
)

plt.plot(
    t, cum_prop_updates_30_MB,
    color='tab:purple', linestyle=(0, (1, 2)), linewidth=2.8,
    label='Proposed: updates only (30 FPS)'
)

# Initial shared-latent marker
plt.scatter(
    [0], [to_MB(z_shared)],
    s=55, color='black', marker='o', zorder=5
)

plt.annotate(
    'Initial shared latent',
    xy=(0, to_MB(z_shared)),
    xytext=(0.35, to_MB(z_shared) * 1.8),
    fontsize=9,
    arrowprops=dict(arrowstyle='->', lw=1, color='black'),
    color='black'
)

# Break-even markers
selected_break_even = [
    ('RGB 30 FPS', t_be_rgb30, rgb, fps1),
    ('Latent 30 FPS', t_be_x030, x0, fps1),
]

for name, t_be, payload, fps in selected_break_even:
    if t_be is not None and t_be <= seconds:
        y_be = to_MB(payload * fps * t_be)
        plt.scatter([t_be], [y_be], s=60, color='black', marker='s', zorder=6)
        plt.annotate(
            f'{name}\n{t_be:.3f} s',
            xy=(t_be, y_be),
            xytext=(t_be + 0.3, y_be + 10),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1, color='black'),
            color='black'
        )

# Summary box
t_end = seconds
rgb30_end = to_MB(rgb * fps1 * t_end)
rgb60_end = to_MB(rgb * fps2 * t_end)
x030_end = to_MB(x0 * fps1 * t_end)
x060_end = to_MB(x0 * fps2 * t_end)
prop30_end = to_MB(z_shared + z_view * fps1 * t_end)

summary_text = (
    f'At {seconds:.0f} s:\n'
    f'RGB 30 FPS = {rgb30_end:.1f} MB\n'
    f'RGB 60 FPS = {rgb60_end:.1f} MB\n'
    f'Latent 30 FPS = {x030_end:.2f} MB\n'
    f'Latent 60 FPS = {x060_end:.2f} MB\n'
    f'Proposed = {prop30_end:.2f} MB'
)

plt.text(
    0.66, 0.28, summary_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.95)
)

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Cumulative transmitted data (MB)', fontsize=12)
plt.title('Cumulative transmitted data vs time under different transmission paradigms', fontsize=14)

plt.yscale('log')
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

plt.legend(
    loc='lower left',
    bbox_to_anchor=(0.18, 0.02),
    fontsize=9,
    framealpha=1.0,
    edgecolor='black'
)

plt.tight_layout()
plt.savefig('cumulative_payload_comparison_richer.pdf', bbox_inches='tight')
plt.savefig('cumulative_payload_comparison_richer.png', dpi=300, bbox_inches='tight')
plt.show()

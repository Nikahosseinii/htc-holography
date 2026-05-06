import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Payload sizes in bytes
# -----------------------------
z_shared = 946176          # one-time shared latent payload
z_view = 256               # per-update view latent payload
rgb = 512 * 512 * 3        # per-frame RGB payload = 786432 bytes
x0 = 4 * (512 // 8) * (512 // 8) * 2   # per-frame diffusion latent payload = 32768 bytes

labels = [
    r'$|z_{\mathrm{shared}}|$' + '\n(one-time)',
    r'$|z_{\mathrm{view}}|$' + '\n(per update)',
    r'$|\mathrm{RGB}|$' + '\n(per frame)',
    r'$|x_0|$' + '\n(per frame)'
]

values = [z_shared, z_view, rgb, x0]

# -----------------------------
# Helper: format byte labels
# -----------------------------
def human_readable(n):
    if n >= 1024**2:
        return f'{n / 1024**2:.2f} MB'
    elif n >= 1024:
        return f'{n / 1024:.2f} KB'
    else:
        return f'{n} B'

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(len(labels))
bars = ax.bar(x, values, width=0.65)

# Use log scale because the payloads differ by orders of magnitude
ax.set_yscale('log')

ax.set_ylabel('Payload size (bytes)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_title('Payload-size breakdown for interactive holographic transmission', fontsize=12)

# Annotate bars
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val * 1.15,
        human_readable(val),
        ha='center',
        va='bottom',
        fontsize=9
    )

# Light grid for readability
ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('payload_size_breakdown.png', dpi=300, bbox_inches='tight')
plt.savefig('payload_size_breakdown.pdf', bbox_inches='tight')
plt.show()

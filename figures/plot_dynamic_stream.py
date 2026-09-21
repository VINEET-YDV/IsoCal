import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Raw sliding-window ECE data, transcribed directly from the
# 3-seed VS Code run (windows at batches 20,40,...,400).
# ============================================================
windows = list(range(20, 401, 20))

data = {
    "tent": {
        0: [14.26,15.92,14.71,14.28,14.95,15.92,14.75,14.23,15.56,16.62,14.59,15.09,16.56,16.78,26.94,26.80,15.86,18.79,21.33,22.20],
        1: [25.87,24.50,24.84,22.50,25.20,23.86,25.20,24.41,25.76,26.71,25.51,26.64,11.76,17.17,16.43,16.54,17.91,15.10,12.49,13.46],
        2: [13.90,10.99,13.49,12.49,13.32,17.14,25.57,27.95,25.33,27.51,26.61,26.82,29.12,25.20,18.72,19.20,17.06,19.75,21.05,19.87],
    },
    "tent_temp": {
        0: [8.27,11.08,10.44,10.53,11.44,12.20,11.70,11.45,12.77,13.97,12.26,12.90,14.43,14.36,23.33,23.58,13.95,16.68,19.20,19.90],
        1: [17.74,17.72,18.90,17.43,20.26,19.40,21.09,20.63,22.15,23.19,22.31,23.57,9.71,14.47,14.04,14.17,15.56,12.99,10.63,11.68],
        2: [8.04,6.55,9.22,8.76,9.92,13.26,20.78,23.51,21.39,23.96,23.24,23.77,26.16,22.47,16.44,17.01,15.08,17.67,18.86,17.83],
    },
    "sicl": {
        0: [6.97,7.16,6.62,6.78,7.25,7.59,8.26,7.53,7.22,8.28,7.27,8.38,8.59,8.37,12.51,10.94,9.75,11.20,12.30,12.08],
        1: [10.00,8.50,9.76,10.64,9.18,9.92,8.96,9.66,11.01,11.58,11.17,11.51,5.71,8.61,8.79,7.04,9.86,7.65,6.73,8.17],
        2: [8.44,6.50,7.51,6.95,7.64,7.91,10.24,11.50,11.15,12.61,13.41,13.77,14.21,12.41,9.52,9.02,8.95,10.65,11.60,11.74],
    },
    "cgt_isocal": {
        0: [7.06,12.06,10.96,10.26,11.15,11.52,10.54,10.17,10.98,11.45,9.34,9.88,11.03,11.20,18.33,14.47,7.10,13.43,14.99,15.32],
        1: [7.74,10.64,11.65,10.99,13.36,12.36,13.30,13.57,13.52,13.38,13.31,13.73,4.31,13.18,10.70,11.88,11.93,9.46,8.74,9.48],
        2: [6.64,8.23,11.26,8.24,10.10,12.82,15.31,13.68,12.50,15.60,14.94,14.61,17.02,12.55,10.66,12.67,10.82,14.15,14.85,13.44],
    },
    "sicl_isocal": {
        0: [3.40,5.04,5.04,4.51,5.47,5.46,5.50,5.12,4.63,6.03,4.96,6.09,5.78,6.88,10.47,8.84,7.14,9.77,10.53,10.90],
        1: [5.35,6.40,6.81,7.25,7.89,6.96,6.79,6.58,6.93,9.25,8.17,9.08,2.92,6.19,6.75,5.84,7.38,6.17,4.33,5.65],
        2: [4.91,4.18,5.97,5.10,4.93,6.25,7.60,9.22,8.40,9.25,9.27,11.09,11.12,9.38,6.78,6.71,6.38,7.26,8.45,8.95],
    },
}

labels = {
    "tent": "TENT",
    "tent_temp": "TENT + temp. scaling (oracle)",
    "cgt_isocal": "cgt_isocal",
    "sicl": "sicl (fused, no isocal)",
    "sicl_isocal": "sicl_isocal (ours)",
}
colors = {
    "tent": "#9e9e9e",
    "tent_temp": "#6b7280",
    "cgt_isocal": "#f2a154",
    "sicl": "#5b9bd5",
    "sicl_isocal": "#d64550",
}

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ------------------------------------------------------------
# Figure 1: bar chart, mean ECE +/- std across all 60 windows
# ------------------------------------------------------------
modes_order = ["tent", "tent_temp", "cgt_isocal", "sicl", "sicl_isocal"]
means, stds = [], []
for m in modes_order:
    vals = np.array(data[m][0] + data[m][1] + data[m][2])
    means.append(vals.mean())
    stds.append(vals.std())

fig, ax = plt.subplots(figsize=(5.2, 3.2))
xpos = np.arange(len(modes_order))
bar_colors = [colors[m] for m in modes_order]
bars = ax.bar(xpos, means, yerr=stds, capsize=4, color=bar_colors,
              edgecolor="black", linewidth=0.6, error_kw={"linewidth": 1})
ax.set_xticks(xpos)
ax.set_xticklabels([labels[m] for m in modes_order], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Mean sliding-window ECE (%)")
ax.set_title("Calibration error under a non-stationary corruption stream\n(CIFAR-10-C, severity 5, 3 seeds, 60 windows)", fontsize=9)
for x, m_val, s_val in zip(xpos, means, stds):
    ax.text(x, m_val + s_val + 0.8, f"{m_val:.1f}", ha="center", fontsize=8)
ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) + 4)
fig.tight_layout()
fig.savefig("/home/claude/fig_ece_bar.png", dpi=300)
fig.savefig("/home/claude/fig_ece_bar.pdf")
plt.close(fig)

# ------------------------------------------------------------
# Figure 2: ECE over the stream, mean across seeds, shaded std
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for m in modes_order:
    arr = np.array([data[m][0], data[m][1], data[m][2]])  # 3 x 20
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)
    ax.plot(windows, mean_curve, label=labels[m], color=colors[m],
            linewidth=2 if m == "sicl_isocal" else 1.3,
            zorder=5 if m == "sicl_isocal" else 3)
    ax.fill_between(windows, mean_curve - std_curve, mean_curve + std_curve,
                     color=colors[m], alpha=0.12, zorder=1)

ax.set_xlabel("Batch index in stream")
ax.set_ylabel("Sliding-window ECE (%)")
ax.set_title("ECE across a 400-batch non-stationary corruption stream\n(mean over 3 seeds, shaded = std)", fontsize=9)
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
ax.set_xlim(20, 400)
fig.tight_layout()
fig.savefig("/home/claude/fig_ece_stream.png", dpi=300)
fig.savefig("/home/claude/fig_ece_stream.pdf")
plt.close(fig)

print("Saved fig_ece_bar.{png,pdf} and fig_ece_stream.{png,pdf}")
print("\nMeans/stds used in bar chart:")
for m, mean_v, std_v in zip(modes_order, means, stds):
    print(f"  {m:12s} mean={mean_v:.2f}  std={std_v:.2f}")

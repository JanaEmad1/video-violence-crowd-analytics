"""Draw the results charts from docs/results.json (numbers taken from the notebook's saved outputs).

    python scripts/make_charts.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})

res = json.loads((DOCS / "results.json").read_text())
h = res["history"]
epochs = np.arange(1, len(h["accuracy"]) + 1)

# 1. Training curves: accuracy and loss as two panels (never a dual axis)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
    ax.plot(epochs, h[key], color=BLUE, lw=2, marker="o", ms=6, label="Train")
    ax.plot(epochs, h["val_" + key], color=ORANGE, lw=2, marker="o", ms=6, label="Validation")
    ax.set(title=title, xlabel="Epoch", xticks=epochs)
    ax.text(epochs[-1] + 0.1, h[key][-1], "train", color=MUTED, va="center", fontsize=9)
    ax.text(epochs[-1] + 0.1, h["val_" + key][-1], "validation", color=MUTED, va="center", fontsize=9)
    ax.set_xlim(0.7, len(epochs) + 0.9)
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle("MobileNet + Bi-LSTM training: the train/validation gap opens after epoch 2",
             fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(DOCS / "training_curves.png", dpi=150)

# 2. Confusion matrix on the 391 held-out videos
y = np.array(res["heldout"]["label"])
p = np.array(res["heldout"]["p_violence"])
pred = np.array(res["heldout"]["pred"])  # labels exactly as the model printed them
cm = np.array([[np.sum((y == t) & (pred == q)) for q in (0, 1)] for t in (0, 1)])
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.3)
names = ["Non-violent", "Violent"]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm[i].sum():.0%} of row", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() * 0.7 else INK, fontsize=12)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=names, yticklabels=names,
       xlabel="Predicted", ylabel="Actual", title=f"Held-out videos (n={len(y)}), threshold 0.5")
ax.grid(False)
fig.tight_layout()
fig.savefig(DOCS / "confusion_matrix.png", dpi=150)

# 3. Alert threshold trade-off: violent clips caught vs. false alarms
thr = np.linspace(0.05, 0.95, 91)
caught = [np.mean(p[y == 1] > t) for t in thr]
false_alarm = [np.mean(p[y == 0] > t) for t in thr]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thr, caught, color=BLUE, lw=2, label="Violent clips caught (recall)")
ax.plot(thr, false_alarm, color=ORANGE, lw=2, label="Calm clips flagged (false-alarm rate)")
for t in (0.3, 0.5):
    i = int(np.argmin(np.abs(thr - t)))
    ax.axvline(t, color=GRAY, lw=1, ls="--")
    ax.text(t + 0.01, 0.47, f"t={t}: catch {caught[i]:.0%},\nfalse alarms {false_alarm[i]:.0%}",
            color=INK, fontsize=9)
ax.set(xlabel="Alert threshold on P(violence)", ylabel="Share of clips", ylim=(0, 1.02),
       title="Choosing the alert threshold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(DOCS / "threshold_tradeoff.png", dpi=150)

# 4. Sample frames strip
frames = [plt.imread(DOCS / f"frame_{i}.png") for i in range(4)]
fig, axes = plt.subplots(1, 4, figsize=(12, 1.75))
for ax, f in zip(axes, frames):
    ink = np.where(f[..., :3].mean(axis=2) < 0.97)  # crop the white figure margin
    ax.imshow(f[ink[0].min():ink[0].max(), ink[1].min():ink[1].max()])
    ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03)
fig.savefig(DOCS / "sample_frames.png", dpi=110)

acc = np.mean(pred == y)
print(f"held-out accuracy {acc:.3f}; confusion {cm.tolist()}")
for t in (0.3, 0.4, 0.5):
    print(t, f"recall {np.mean(p[y == 1] > t):.3f}", f"false alarms {np.mean(p[y == 0] > t):.3f}")

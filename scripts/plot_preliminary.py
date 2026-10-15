"""Charts for the preliminary results (grayscale + hatch, print-safe)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/preliminary")
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#555",
        "axes.linewidth": 0.8,
    }
)

s = pd.read_csv(OUT / "summary.csv")
s = s[s.Setup == "calibrated"].set_index("Arm")
arms = ["Alert-only", "Always-retrain", "Threshold rules"]
harm = [s.loc[a, "Harmful action rate"] * 100 for a in arms]
corr = [s.loc[a, "Correct action rate"] * 100 for a in arms]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = range(len(arms))
w = 0.36
b1 = ax.bar([i - w / 2 - 0.01 for i in x], harm, w, color="#3a3a3a", label="Harmful action rate")
b2 = ax.bar(
    [i + w / 2 + 0.01 for i in x],
    corr,
    w,
    color="white",
    edgecolor="#3a3a3a",
    hatch="///",
    linewidth=1.0,
    label="Correct action rate",
)
for bars in (b1, b2):
    for r in bars:
        ax.text(
            r.get_x() + r.get_width() / 2,
            r.get_height() + 1.5,
            f"{r.get_height():.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222",
        )
ax.set_xticks(list(x), arms)
ax.set_ylim(0, 110)
ax.set_ylabel("% of episodes")
ax.yaxis.grid(True, color="#e3e3e3", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Elec2, 75 injected episodes (calibrated thresholds)", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig_arms_comparison.png", dpi=220)
plt.close(fig)

rep = json.loads((OUT / "report.json").read_text())
fa = rep["default_false_alarms_on_clean_windows"]
labels = list(fa)
vals = [fa[k] * 100 for k in labels]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
bars = ax.barh(labels[::-1], vals[::-1], color="#3a3a3a", height=0.55)
for r in bars:
    ax.text(
        r.get_width() + 1.5,
        r.get_y() + r.get_height() / 2,
        f"{r.get_width():.0f}%",
        va="center",
        fontsize=10,
        color="#222",
    )
ax.set_xlim(0, 112)
ax.set_xlabel("% of clean windows flagged (no fault injected)")
ax.xaxis.grid(True, color="#e3e3e3", linewidth=0.7)
ax.set_axisbelow(True)
ax.set_title("Off-the-shelf drift thresholds on clean Elec2 data", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig_false_alarms.png", dpi=220)
plt.close(fig)
print("saved")

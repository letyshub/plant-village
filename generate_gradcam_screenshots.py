"""Generate Grad-CAM comparison images for blog post."""

import sys
sys.path.insert(0, "/mnt/d/Programs/plant-village")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from src.predict import PlantDiseasePredictor

MODEL_PATH = "outputs/models/plant_disease_model.h5"
CLASSES_PATH = "outputs/models/plant_disease_model_classes.txt"
OUT_DIR = Path("blog/images")

predictor = PlantDiseasePredictor(MODEL_PATH, CLASSES_PATH)

IMAGES = [
    ("blog/images/tomato_healthy.jpg",      "Zdrowy liść pomidora",       "Healthy Tomato Leaf"),
    ("blog/images/tomato_late_blight.jpg",  "Pomidor — Zaraza późna",     "Tomato — Late Blight"),
    ("blog/images/apple_healthy.jpg",        "Zdrowy liść jabłoni",        "Healthy Apple Leaf"),
    ("blog/images/apple_scab.jpg",           "Jabłoń — Parch jabłoni",     "Apple — Apple Scab"),
]

def format_label(raw: str) -> str:
    """Turn class name like Apple___Apple_scab into readable string."""
    raw = raw.replace("_", " ").strip()
    if "___" in raw:
        plant, disease = raw.split("___", 1)
        return f"{plant.strip()} — {disease.strip()}"
    return raw


def make_panel(img_path: str, title_pl: str, title_en: str) -> dict:
    img_bgr = cv2.imread(img_path)
    label, conf, heatmap_bgr = predictor.predict(img_bgr)

    orig_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    heat_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    top5 = predictor.top_k(img_bgr, k=5)

    return {
        "orig": orig_rgb,
        "heat": heat_rgb,
        "label": format_label(label),
        "conf": conf,
        "top5": [(format_label(n), c) for n, c in top5],
        "title_pl": title_pl,
        "title_en": title_en,
    }


panels = [make_panel(*args) for args in IMAGES]


# ── Figure 1: 2×2 grid — original vs Grad-CAM ──────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(12, 20))
fig.patch.set_facecolor("#1e1e2e")

for row, p in enumerate(panels):
    for col, (img, subtitle) in enumerate([(p["orig"], "Input"), (p["heat"], "Grad-CAM")]):
        ax = axes[row][col]
        ax.imshow(img)
        ax.set_title(
            f"{p['title_en']}  [{subtitle}]" if col == 0
            else f"{p['label']}  {p['conf']:.1%}",
            color="white", fontsize=10, pad=6
        )
        ax.axis("off")

plt.suptitle("Plant Disease Classifier — Grad-CAM Visualization", color="white",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out1 = OUT_DIR / "gradcam_grid.png"
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out1}")


# ── Figure 2: single detailed panel — tomato late blight ───────────────────
p = panels[1]  # tomato late blight

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor("#1e1e2e")
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.05)

# Original
ax0 = fig.add_subplot(gs[0])
ax0.imshow(p["orig"])
ax0.set_title("Input image", color="white", fontsize=11)
ax0.axis("off")

# Grad-CAM
ax1 = fig.add_subplot(gs[1])
ax1.imshow(p["heat"])
ax1.set_title("Grad-CAM heatmap", color="white", fontsize=11)
ax1.axis("off")

# Top-5 bar chart
ax2 = fig.add_subplot(gs[2])
ax2.set_facecolor("#2a2a3e")
names = [n[:28] + "…" if len(n) > 28 else n for n, _ in reversed(p["top5"])]
confs = [c for _, c in reversed(p["top5"])]
colors = ["#e74c3c" if i == len(confs) - 1 else "#5dade2" for i in range(len(confs))]
bars = ax2.barh(names, confs, color=colors, height=0.5)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("Confidence", color="white")
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[:].set_color("#555")
for bar, val in zip(bars, confs):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.1%}", va="center", color="white", fontsize=8)
ax2.set_title("Top-5 predictions", color="white", fontsize=11)

fig.suptitle(
    f"Prediction: {p['label']}   Confidence: {p['conf']:.1%}",
    color="#e74c3c", fontsize=13, fontweight="bold"
)
plt.tight_layout()
out2 = OUT_DIR / "gradcam_detail_tomato_blight.png"
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out2}")


# ── Figure 3: apple scab detail ─────────────────────────────────────────────
p = panels[3]  # apple scab

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor("#1e1e2e")
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.05)

ax0 = fig.add_subplot(gs[0])
ax0.imshow(p["orig"])
ax0.set_title("Input image", color="white", fontsize=11)
ax0.axis("off")

ax1 = fig.add_subplot(gs[1])
ax1.imshow(p["heat"])
ax1.set_title("Grad-CAM heatmap", color="white", fontsize=11)
ax1.axis("off")

ax2 = fig.add_subplot(gs[2])
ax2.set_facecolor("#2a2a3e")
names = [n[:28] + "…" if len(n) > 28 else n for n, _ in reversed(p["top5"])]
confs = [c for _, c in reversed(p["top5"])]
colors = ["#e74c3c" if i == len(confs) - 1 else "#5dade2" for i in range(len(confs))]
bars = ax2.barh(names, confs, color=colors, height=0.5)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("Confidence", color="white")
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[:].set_color("#555")
for bar, val in zip(bars, confs):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.1%}", va="center", color="white", fontsize=8)
ax2.set_title("Top-5 predictions", color="white", fontsize=11)

fig.suptitle(
    f"Prediction: {p['label']}   Confidence: {p['conf']:.1%}",
    color="#e74c3c", fontsize=13, fontweight="bold"
)
plt.tight_layout()
out3 = OUT_DIR / "gradcam_detail_apple_scab.png"
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out3}")

print("\nAll done.")

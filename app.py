"""
Gradio demo — Plant Disease Detector.

Run:
    python app.py
    python app.py --model model/plant_disease_model.h5 --classes model/plant_disease_model_classes.txt
"""

import argparse
import os

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.predict import PlantDiseasePredictor

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models/plant_disease_model.h5")
CLASSES_PATH = os.getenv("CLASSES_PATH", "outputs/models/plant_disease_model_classes.txt")

_predictor: PlantDiseasePredictor | None = None


def get_predictor() -> PlantDiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PlantDiseasePredictor(MODEL_PATH, CLASSES_PATH)
    return _predictor


def diagnose(image: np.ndarray) -> tuple[str, Image.Image, str]:
    """
    Gradio handler: receives RGB numpy array, returns
    (diagnosis_text, grad_cam_image, top5_markdown).
    """
    if image is None:
        return "No image provided.", None, ""

    predictor = get_predictor()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    label, confidence, heatmap_bgr = predictor.predict(bgr)
    top5 = predictor.top_k(bgr, k=5)

    # Format diagnosis
    plant, *disease_parts = label.replace("_", " ").split("___")
    disease = disease_parts[0] if disease_parts else "healthy"
    diagnosis = f"**{plant}** — {disease}\nConfidence: **{confidence:.1%}**"

    # Convert heatmap BGR → PIL RGB
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Top-5 markdown table
    rows = "\n".join(
        f"| {i+1} | {name.replace('_', ' ')} | {conf:.1%} |"
        for i, (name, conf) in enumerate(top5)
    )
    top5_md = f"| # | Class | Confidence |\n|---|-------|------------|\n{rows}"

    return diagnosis, heatmap_pil, top5_md


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Plant Disease Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Plant Disease Detector\nUpload a leaf photo to diagnose the disease.")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Leaf photo", type="numpy")
                submit_btn = gr.Button("Diagnose", variant="primary")
            with gr.Column():
                diagnosis_out = gr.Markdown(label="Diagnosis")
                gradcam_out = gr.Image(label="Grad-CAM heatmap")
                top5_out = gr.Markdown(label="Top-5 predictions")

        submit_btn.click(
            fn=diagnose,
            inputs=image_input,
            outputs=[diagnosis_out, gradcam_out, top5_out],
        )

        gr.Markdown(
            "Model: EfficientNetB0 fine-tuned on [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) "
            "— 38 classes, 54 000+ images."
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--classes", default=CLASSES_PATH)
    parser.add_argument("--share", action="store_true", help="Create public HuggingFace link")
    args = parser.parse_args()

    MODEL_PATH = args.model
    CLASSES_PATH = args.classes

    demo = build_ui()
    demo.launch(share=args.share)

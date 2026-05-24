"""Flask API for the insect classifier, served with OpenVINO inference.

One endpoint:

    POST /predict   multipart/form-data with field 'image'  ->  JSON top-5

On first startup the PyTorch checkpoint (checkpoints/best.pt) is converted to
OpenVINO IR (checkpoints/best.xml + best.bin) and cached. Subsequent boots
load the IR directly. Inference runs on the OpenVINO CPU plugin by default —
typically 3-5x faster than PyTorch CPU and competitive with Intel iGPU. Switch
DEVICE to "GPU" if you have an Intel iGPU/dGPU.

Install + run:

    .venv/bin/python -m pip install flask openvino
    .venv/bin/python www/app.py

Test:

    curl -F "image=@/path/to/insect.jpg" http://127.0.0.1:5000/predict
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
IR_XML = CKPT_DIR / "best.xml"
DEVICE = "CPU"  # "GPU" if you have an Intel iGPU/dGPU; NVIDIA users keep "CPU"


def export_to_openvino(num_classes: int) -> None:
    """Convert checkpoints/best.pt -> best.xml + best.bin (one-time, cached on disk)."""
    print(f"[export] converting best.pt -> {IR_XML.name} (num_classes={num_classes})")
    model = efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(
        torch.load(CKPT_DIR / "best.pt", map_location="cpu", weights_only=True)
    )
    model.train(False)
    # Trace with batch=2 (orig + flip TTA) but mark batch dim as dynamic afterwards
    # so the model also accepts batch=1 or larger if you ever change the policy.
    example = torch.randn(2, 3, 380, 380)
    ov_model = ov.convert_model(model, example_input=example)
    ov_model.reshape([-1, 3, 380, 380])
    ov.save_model(ov_model, IR_XML)
    print(f"[export] saved {IR_XML} + {IR_XML.with_suffix('.bin')}")


def load_inference():
    """Load class maps + ensure IR exists + compile for the target device."""
    label_to_idx = json.load(open(CKPT_DIR / "label_to_idx.json"))
    idx_to_taxon = {v: int(k) for k, v in label_to_idx.items()}

    counts = pd.read_csv(ROOT / "class_counts.csv")
    taxon_to_name = dict(zip(counts["taxon_id"].astype(int), counts["scientific_name"]))

    if not IR_XML.exists():
        export_to_openvino(len(label_to_idx))

    core = ov.Core()
    print(f"[startup] OpenVINO available devices: {core.available_devices}")
    compiled = core.compile_model(core.read_model(IR_XML), DEVICE)
    transform = EfficientNet_B4_Weights.DEFAULT.transforms()
    return compiled, transform, idx_to_taxon, taxon_to_name, len(label_to_idx)


app = Flask(__name__)
COMPILED, TRANSFORM, IDX_TO_TAXON, TAXON_TO_NAME, NUM_CLASSES = load_inference()
OUTPUT_KEY = COMPILED.output(0)
print(f"[startup] OpenVINO model ready on {DEVICE} with {NUM_CLASSES} classes")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' form field"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot decode image: {e}"}), 400

    # Identical preprocessing to training (the .transforms() from the weights bundle).
    # Output is a torch tensor; convert to numpy for OpenVINO. Then stack the
    # horizontal flip alongside for free test-time augmentation.
    x = TRANSFORM(img).unsqueeze(0).numpy()                    # (1, 3, 380, 380)
    batch = np.concatenate([x, x[:, :, :, ::-1].copy()], 0)    # (2, 3, 380, 380)

    logits = COMPILED(batch)[OUTPUT_KEY]                       # (2, num_classes)
    probs = softmax(logits, axis=1).mean(axis=0)               # average over orig + flip
    top5 = probs.argsort()[::-1][:5]

    predictions = []
    for idx in top5:
        taxon_id = IDX_TO_TAXON[int(idx)]
        predictions.append({
            "taxon_id": taxon_id,
            "scientific_name": TAXON_TO_NAME.get(taxon_id, "unknown"),
            "score": float(probs[int(idx)]),
        })
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

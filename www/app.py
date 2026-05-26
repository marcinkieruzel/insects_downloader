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
from flask import Flask, jsonify, render_template_string, request
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


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Insect classifier</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { margin-bottom: 0.2em; }
  .sub { color: #666; margin-top: 0; }
  form { margin: 1.5em 0; display: flex; gap: 0.6em; align-items: center; }
  button { padding: 0.5em 1em; cursor: pointer; }
  #preview { max-width: 100%; max-height: 320px; margin-top: 1em; border: 1px solid #ddd; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; margin-top: 1em; }
  th, td { padding: 0.5em 0.6em; border-bottom: 1px solid #eee; text-align: left; }
  .score { text-align: right; font-variant-numeric: tabular-nums; }
  .err { color: #b00; margin-top: 1em; }
  .muted { color: #888; }
  .sci { font-style: italic; }
</style>
</head>
<body>
<h1>Insect classifier</h1>
<p class="sub">{{ num_classes }} European species &middot; OpenVINO on {{ device }}</p>

<form id="f">
  <input type="file" name="image" accept="image/*" required>
  <button type="submit">Predict</button>
</form>

<img id="preview" hidden>
<div id="out"></div>

<script>
const form = document.getElementById('f');
const out = document.getElementById('out');
const preview = document.getElementById('preview');

function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
function el(tag, opts) {
  const n = document.createElement(tag);
  if (opts && opts.className) n.className = opts.className;
  if (opts && opts.text != null) n.textContent = String(opts.text);
  return n;
}
function message(text, cls) {
  clear(out);
  out.appendChild(el('p', { className: cls || 'muted', text: text }));
}

function renderPredictions(predictions) {
  clear(out);
  const table = el('table');
  const thead = el('thead');
  const hr = el('tr');
  hr.appendChild(el('th', { text: 'Species' }));
  hr.appendChild(el('th', { className: 'score', text: 'Score' }));
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = el('tbody');
  predictions.forEach(p => {
    const tr = el('tr');
    const tdName = el('td');
    tdName.appendChild(el('span', { className: 'sci', text: p.scientific_name }));
    tdName.appendChild(document.createTextNode(' '));
    tdName.appendChild(el('span', { className: 'muted', text: '(taxon ' + p.taxon_id + ')' }));
    tr.appendChild(tdName);
    tr.appendChild(el('td', { className: 'score', text: (p.score * 100).toFixed(1) + '%' }));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  out.appendChild(table);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = form.image.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  message('Predicting…', 'muted');
  const data = new FormData();
  data.append('image', file);
  try {
    const r = await fetch('/predict', { method: 'POST', body: data });
    const j = await r.json();
    if (!r.ok) { message(j.error || ('HTTP ' + r.status), 'err'); return; }
    renderPredictions(j.predictions);
  } catch (err) {
    message(err.message, 'err');
  }
});
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML, num_classes=NUM_CLASSES, device=DEVICE)


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

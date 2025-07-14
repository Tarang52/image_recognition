"""
Flask + TensorFlow API — Render-ready (no index.html served)
Lazy-loads the model on first request so the container boots fast.
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# -------------------------------------------------------------------- CONFIG
MODEL_PATH   = "checkpoints/mobilenetv3_enhanced/mobilenetv3_enhanced.keras"
LABELS_PATH  = "checkpoints/mobilenetv3_enhanced/labels.txt"
UPLOADS_DIR  = "uploads"
IMG_SIZE     = (224, 224)

# -------------------------------------------------------------------- INIT
app = Flask(__name__)
CORS(app)                                   # allow Vercel origin
os.makedirs(UPLOADS_DIR, exist_ok=True)      # survive empty-dir purges

model = None
CLASS_NAMES = []

def lazy_load_model():
    """Load model & labels only once (on first /predict)."""
    global model, CLASS_NAMES
    if model is None:
        print("🔁 Lazy-loading TensorFlow model …")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            CLASS_NAMES = f.read().splitlines()
        print("✅ Model ready. Classes:", CLASS_NAMES)

# -------------------------------------------------------------------- ROUTES
@app.route("/", methods=["GET"])
def health():
    """Simple health-check for uptime monitors."""
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    lazy_load_model()                       # ensure model is loaded

    if "file" not in request.files:
        return jsonify(error="No file provided"), 400

    file_obj = request.files["file"]
    if file_obj.filename == "":
        return jsonify(error="Empty filename"), 400

    tmp_path = Path(UPLOADS_DIR) / file_obj.filename
    file_obj.save(tmp_path)

    try:
        # image → array
        img = image.load_img(tmp_path, target_size=IMG_SIZE)
        arr = image.img_to_array(img)
        arr = preprocess_input(arr)
        batch = np.expand_dims(arr, 0)

        # inference
        preds = model.predict(batch)
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx]) * 100

        return {
            "class": CLASS_NAMES[idx],
            "confidence": f"{conf:.2f}"
        }
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=str(e)), 500
    finally:
        # optional: clean temp image
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass

# -------------------------------------------------------------------- LOCAL DEV
if __name__ == "__main__":                   # ignored by Gunicorn on Render
    app.run(debug=True, port=8080)

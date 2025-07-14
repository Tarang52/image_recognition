"""
app.py – Flask + MobileNetV3 API
Works locally (python app.py) and on Railway with Procfile:  web: gunicorn app:app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import os
from pathlib import Path
import traceback

# ───── CONFIG ─────
MODEL_PATH  = Path("checkpoints/mobilenetv3_enhanced/mobilenetv3_enhanced.keras")
LABELS_PATH = Path("checkpoints/mobilenetv3_enhanced/labels.txt")
UPLOAD_DIR  = Path("uploads")
IMG_SIZE    = (224, 224)

# ───── INIT ─────
app = Flask(__name__, static_folder=".", static_url_path="/")
CORS(app)
UPLOAD_DIR.mkdir(exist_ok=True)

# ───── LOAD MODEL ─────
print("🔁  Loading model…")
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found → {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found → {LABELS_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    CLASS_NAMES = LABELS_PATH.read_text().splitlines()
    print("✅  Model & labels loaded. Classes:", CLASS_NAMES)
except Exception as e:
    print("❌  Failed to load model:", e)
    model, CLASS_NAMES = None, []
    # We keep running so /health still works

# ───── ROUTES ─────
@app.route("/")
def index():
    """Serve the single-page app (index.html)."""
    return app.send_static_file("index.html")

@app.route("/health")
def health():
    """Railway pings this to check liveness."""
    status = "up" if model else "model_load_failed"
    return jsonify({"status": status})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No image file provided"}), 400

    file_obj = request.files["file"]
    file_path = UPLOAD_DIR / file_obj.filename
    file_obj.save(file_path)

    try:
        # ── preprocess ──
        img = image.load_img(file_path, target_size=IMG_SIZE)
        x   = preprocess_input(image.img_to_array(img))
        x   = np.expand_dims(x, 0)

        # ── predict ──
        probs  = model.predict(x)[0]
        idx    = int(np.argmax(probs))
        label  = CLASS_NAMES[idx]
        conf   = float(probs[idx]) * 100

        return jsonify({"class": label, "confidence": f"{conf:.2f}"})
    except Exception as err:
        print("❌  PREDICT ERROR:", err)
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500
    finally:
        # Delete the uploaded image to save disk space
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass

# ───── LOCAL DEV ENTRYPOINT ─────
if __name__ == "__main__":
    print("🧪  Running with Flask dev server (localhost:5000)")
    app.run(debug=True, port=5000)

"""
app.py – Flask + MobileNetV3 API
Run locally:           python app.py
Run on Railway:        Procfile →  web: gunicorn app:app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import os, traceback, sys
from pathlib import Path

# ───── CONFIG ─────
MODEL_PATH  = Path("checkpoints/mobilenetv3_enhanced/mobilenetv3_enhanced.keras")
LABELS_PATH = Path("checkpoints/mobilenetv3_enhanced/labels.txt")
UPLOAD_DIR  = Path("uploads")
IMG_SIZE    = (224, 224)
PORT        = int(os.getenv("PORT", 5000))

# ───── INIT ─────
app = Flask(__name__, static_folder=".", static_url_path="/")
CORS(app)
UPLOAD_DIR.mkdir(exist_ok=True)

print("🚀 Booting Flask – cwd:", os.getcwd(), flush=True)

# ───── LOAD MODEL ─────
model, CLASS_NAMES = None, []
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file missing → {MODEL_PATH.resolve()}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"❌ Labels file missing → {LABELS_PATH.resolve()}")

    print("🔁 Loading model…", flush=True)
    model = tf.keras.models.load_model(str(MODEL_PATH))
    CLASS_NAMES = LABELS_PATH.read_text().splitlines()

    print(f"✅ Model ready with {len(CLASS_NAMES)} classes {CLASS_NAMES}", flush=True)
except Exception as e:
    # Keep server up so /health explains the problem
    print("🚨 Model load failed:", e, file=sys.stderr, flush=True)

# ───── ROUTES ─────
@app.route("/")
def root():
    """Serve SPA frontend."""
    return app.send_static_file("index.html")

@app.route("/health")
def health():
    """Simple liveness check for Railway / uptime robot."""
    return jsonify({"status": "up" if model else "model_load_failed"}), (200 if model else 500)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "no image file provided"}), 400

    # save upload
    up_file = request.files["file"]
    tmp_path = UPLOAD_DIR / up_file.filename
    up_file.save(tmp_path)

    try:
        # preprocess
        img = image.load_img(tmp_path, target_size=IMG_SIZE)
        x   = preprocess_input(image.img_to_array(img))
        x   = np.expand_dims(x, 0)

        # inference
        probs = model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf  = float(probs[idx]) * 100

        return jsonify({"class": label, "confidence": f"{conf:.2f}"})
    except Exception as err:
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500
    finally:
        # tidy up disk
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# ───── LOCAL DEV ENTRYPOINT ─────
if __name__ == "__main__":
    print(f"🧪  Local dev server at http://127.0.0.1:{PORT}", flush=True)
    app.run(debug=True, port=PORT)

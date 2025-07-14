from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pathlib import Path

# ───── CONFIG ─────
MODEL_PATH = "checkpoints/mobilenetv3_enhanced/mobilenetv3_enhanced.keras"
LABELS_PATH = "checkpoints/mobilenetv3_enhanced/labels.txt"
UPLOAD_FOLDER = "uploads"
IMG_SIZE = (224, 224)

# ───── INIT ─────
app = Flask(__name__, static_folder=".", static_url_path="/")
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# ───── LOAD MODEL + LABELS ─────
print("🔁 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        CLASS_NAMES = f.read().splitlines()
    print("✅ Model loaded. Classes:", CLASS_NAMES)
except Exception as e:
    print("❌ Model load failed:", e)
    model = None
    CLASS_NAMES = []

# ───── ROUTES ─────
@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # Load and preprocess image
        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_batch)
        pred_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_idx]) * 100
        label = CLASS_NAMES[pred_idx]

        return jsonify({
            "class": label,
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ───── LOCAL DEV ONLY ─────
if __name__ == "__main__":
    print("🧪 Running locally with Flask dev server.")
    app.run(debug=True)

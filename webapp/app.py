import os
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from flask import Flask, render_template, request, redirect, flash
import torch
import torch.nn.functional as F
from torchvision import transforms

from network import Network  # your CNN architecture

# Configuration
BASE_DIR = Path(__file__).resolve().parent
HF_MODEL_URL = "https://huggingface.co/AurevinP/Cat_vs_Dog_cnn/resolve/main/cat_v_dog_cnn.pth"
MODEL_PATH = BASE_DIR / "cat_v_dog_cnn.pth"
DEVICE = torch.device("cpu")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-string"  # change for production

# Preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

CLASS_MAP = {0: "cat", 1: "dog"}

# -----------------------------
# Download model from Hugging Face if not present
# -----------------------------
if not MODEL_PATH.exists():
    print("Downloading model from Hugging Face...")
    resp = requests.get(HF_MODEL_URL, timeout=60)
    resp.raise_for_status()
    MODEL_PATH.write_bytes(resp.content)
    print("Model download complete.")

# -----------------------------
# Load and prepare model
# -----------------------------
model = Network()
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# Warm up to reduce first-request latency
with torch.no_grad():
    dummy = torch.zeros(1, 3, 200, 200).to(DEVICE)
    model(dummy)

# -----------------------------
# Helper functions
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img: Image.Image):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        probs = F.softmax(out, dim=1)
        prob_values = probs.cpu().numpy()[0]
        predicted_class = int(prob_values.argmax())
        predicted_prob = float(prob_values[predicted_class])
        label = CLASS_MAP.get(predicted_class, str(predicted_class))
        return label, predicted_prob

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            img = Image.open(file.stream).convert("RGB")
            label, prob = predict_image(img)
            return render_template("index.html", filename=None, label=label, prob=prob * 100)
        else:
            flash("Allowed image types: png, jpg, jpeg, bmp")
            return redirect(request.url)

    return render_template("index.html", filename=None)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

import os
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from network import Network  # use your Network from network.py

# Configuration
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MODEL_PATH = BASE_DIR / "cat_v_dog_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB
app.secret_key = "replace-with-a-secure-random-string"  # change for production

# Preprocessing (must match training)
transform = transforms.Compose(
    [
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Class mapping (keep consistent with training)
CLASS_MAP = {0: "cat", 1: "dog"}

# Load model once
model = Network()
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train and save model as cat_v_dog_cnn.pth")
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
            filename = secure_filename(file.filename)
            save_path = UPLOAD_FOLDER / filename

            # Ensure unique filename to avoid overwrite
            counter = 1
            stem = save_path.stem
            while save_path.exists():
                save_path = UPLOAD_FOLDER / f"{stem}_{counter}{save_path.suffix}"
                counter += 1

            file.save(save_path)

            # Predict
            label, prob = predict_image(save_path)
            # prob is in [0,1]
            return render_template(
                "index.html",
                filename=save_path.name,
                label=label,
                prob=prob*100,  # Pass as float percentage (0-100)
            )
        else:
            flash("Allowed image types: png, jpg, jpeg, bmp")
            return redirect(request.url)

    return render_template("index.html", filename=None)


def predict_image(image_path: Path):
    """
    Loads image from disk, preprocesses, runs model and returns label & prob.
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)  # shape (1,3,200,200)

    with torch.no_grad():
        out = model(x)  # shape (1, num_classes)
        # Ensure out is (1,2)
        if out.dim() == 1:
            # single-dim, expand
            out = out.unsqueeze(0)
        probs = F.softmax(out, dim=1)
        prob_values = probs.cpu().numpy()[0]  # e.g. [0.2, 0.8]
        predicted_class = int(prob_values.argmax())
        predicted_prob = float(prob_values[predicted_class])
        label = CLASS_MAP.get(predicted_class, str(predicted_class))
        return label, predicted_prob


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # For development only.
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from gradcam import generate_gradcam
from PIL import Image
import io

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load model
MODEL_PATH = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load InceptionV3
model = models.inception_v3(pretrained=False)
model.fc = nn.Linear(2048, 2)  # Pneumonia: Binary Classification
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image Preprocessing
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    input_tensor = transform_image(image_bytes)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    result = "Pneumonia Detected" if predicted_class.item() == 1 else "Normal"

    # Generate Grad-CAM
    gradcam_image_path = generate_gradcam(model, input_tensor, predicted_class.item())

    return jsonify({
        "prediction": result,
        "confidence": float(torch.softmax(output, dim=1)[0][predicted_class.item()]),
        "gradcam_url": f"/gradcam?file={gradcam_image_path}"
    })

@app.route("/gradcam", methods=["GET"])
def get_gradcam():
    file_path = request.args.get("file")
    return send_file(file_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
        

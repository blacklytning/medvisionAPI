# import os
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torchvision.models as models
# from flask import Flask, request, jsonify, send_file
# from PIL import Image
# from flask_cors import CORS
#
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes
#
# # Model path and class names
# MODEL_PATH = "model/model.pth"
# CLASS_NAMES = ["Normal", "Pneumonia"]
#
# # Load the trained model
# def load_model(model_path):
#     model = models.inception_v3(aux_logits=True)
#     n_features = model.fc.in_features
#     model.fc = nn.Linear(n_features, len(CLASS_NAMES))
#
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     return model
#
# # Initialize model
# model = load_model(MODEL_PATH)
#
# # Define image transformations
# def transform_image(image):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((299, 299)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.4822, 0.4822, 0.4822], [0.2362, 0.2362, 0.2362]),
#         ]
#     )
#     return transform(image).unsqueeze(0)
#
# # Prediction function
# def predict(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = transform_image(image)
#
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted_class = torch.max(output, 1)
#         confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()
#
#     return CLASS_NAMES[predicted_class.item()], confidence, image_tensor
#
# # Grad-CAM logic
# def generate_gradcam(model, image_tensor, target_class):
#     gradients = None
#     activations = None
#
#     def backward_hook(module, grad_input, grad_output):
#         nonlocal gradients
#         gradients = grad_output[0]
#
#     def forward_hook(module, input, output):
#         nonlocal activations
#         activations = output
#
#     # Get the last convolutional layer (Mixed_7c/Inception)
#     target_layer = model.Mixed_7c
#     forward_handle = target_layer.register_forward_hook(forward_hook)
#     backward_handle = target_layer.register_full_backward_hook(backward_hook)
#
#     # Forward pass
#     output = model(image_tensor)
#     class_score = output[0, target_class]
#
#     # Backprop to get gradients
#     model.zero_grad()
#     class_score.backward(retain_graph=True)
#
#     # Remove hooks
#     forward_handle.remove()
#     backward_handle.remove()
#
#     # Generate Grad-CAM
#     gradients = gradients.detach().cpu().numpy()
#     activations = activations.detach().cpu().numpy()
#
#     weights = np.mean(gradients, axis=(2, 3))  # Global Average Pooling
#     gradcam = np.sum(weights[:, :, None, None] * activations, axis=1)
#     gradcam = np.maximum(gradcam, 0)  # ReLU activation to remove negatives
#     gradcam /= np.max(gradcam)  # Normalize heatmap
#
#     return gradcam[0]
#
# # Overlay Grad-CAM on original image
# def overlay_gradcam_on_image(img_path, gradcam, output_path, alpha=0.5):
#     heatmap = gradcam
#     heatmap = cv2.resize(heatmap, (299, 299))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (299, 299))
#
#     overlay = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
#     cv2.imwrite(output_path, overlay)
#
# # API route for predictions with Grad-CAM
# @app.route("/predict", methods=["POST"])
# def predict_image():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400
#
#     # Save the file to the uploads folder
#     filepath = os.path.join("uploads", file.filename)
#     file.save(filepath)
#
#     try:
#         label, confidence, image_tensor = predict(filepath)
#         predicted_class = CLASS_NAMES.index(label)
#
#         # Generate Grad-CAM and overlay on original image
#         gradcam = generate_gradcam(model, image_tensor, predicted_class)
#         gradcam_output_path = os.path.join("uploads", "gradcam_" + file.filename)
#         overlay_gradcam_on_image(filepath, gradcam, gradcam_output_path)
#
#         os.remove(filepath)  # Delete original file after prediction
#         return jsonify(
#             {
#                 "prediction": label,
#                 "confidence": round(confidence * 100, 2),
#                 "gradcam_url": f"/gradcam?file=gradcam_{file.filename}",
#             }
#         )
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # API route to serve Grad-CAM image
# @app.route("/gradcam", methods=["GET"])
# def get_gradcam():
#     file_name = request.args.get("file")
#     file_path = os.path.join("uploads", file_name)
#
#     if os.path.exists(file_path):
#         return send_file(file_path, mimetype="image/png")
#     else:
#         return jsonify({"error": "File not found"}), 404
#
# # Home route
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Welcome to Pneumonia Detection API"})
#
# # Run Flask app
# if __name__ == "__main__":
#     os.makedirs("uploads", exist_ok=True)
#     app.run(host="0.0.0.0", port=5000)

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
from PIL import Image
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model path and class names
MODEL_PATH = "model/model.pth"
CLASS_NAMES = ["Normal", "Pneumonia"]

# Load the trained model
def load_model(model_path):
    model = models.inception_v3(aux_logits=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, len(CLASS_NAMES))

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Initialize model
model = load_model(MODEL_PATH)

# Define image transformations
def transform_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.4822, 0.4822, 0.4822], [0.2362, 0.2362, 0.2362]),
        ]
    )
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

    return CLASS_NAMES[predicted_class.item()], confidence, image_tensor

# Grad-CAM logic
def generate_gradcam(model, image_tensor, target_class):
    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # Get the last convolutional layer (Mixed_7c/Inception)
    target_layer = model.Mixed_7c
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    class_score = output[0, target_class]

    # Backprop to get gradients
    model.zero_grad()
    class_score.backward(retain_graph=True)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Generate Grad-CAM
    gradients = gradients.detach().cpu().numpy()
    activations = activations.detach().cpu().numpy()

    weights = np.mean(gradients, axis=(2, 3))  # Global Average Pooling
    gradcam = np.sum(weights[:, :, None, None] * activations, axis=1)
    gradcam = np.maximum(gradcam, 0)  # ReLU activation to remove negatives
    gradcam /= np.max(gradcam)  # Normalize heatmap

    return gradcam[0]

# Overlay Grad-CAM on original image
def overlay_gradcam_on_image(img_path, gradcam, output_path, alpha=0.5):
    heatmap = gradcam
    heatmap = cv2.resize(heatmap, (299, 299))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))

    overlay = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    cv2.imwrite(output_path, overlay)

# API route for predictions with Grad-CAM
@app.route("/predict", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Generate a unique identifier for each image
    unique_id = str(uuid.uuid4().hex[:8])
    filename = f"{unique_id}_{file.filename}"
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    try:
        label, confidence, image_tensor = predict(filepath)
        predicted_class = CLASS_NAMES.index(label)

        # Generate Grad-CAM and overlay on original image
        gradcam = generate_gradcam(model, image_tensor, predicted_class)
        gradcam_filename = f"gradcam_{filename}"
        gradcam_output_path = os.path.join("uploads", gradcam_filename)
        overlay_gradcam_on_image(filepath, gradcam, gradcam_output_path)

        os.remove(filepath)  # Delete original file after prediction

        # Return the result with corrected URL
        return jsonify(
            {
                "id": unique_id,
                "filename": file.filename,
                "prediction": label,
                "confidence": round(confidence * 100, 2),
                "gradcam_url": url_for("get_gradcam", file=gradcam_filename, _external=True),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API route to serve Grad-CAM image
@app.route("/gradcam", methods=["GET"])
def get_gradcam():
    file_name = request.args.get("file")
    file_path = os.path.join("uploads", file_name)

    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/png")
    else:
        return jsonify({"error": "File not found"}), 404

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Pneumonia Detection API"})

# Run Flask app
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)


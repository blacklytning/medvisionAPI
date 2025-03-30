import cv2
import numpy as np
import torch
import torch.nn.functional as F

def generate_gradcam(model, input_tensor, class_idx):
    model.eval()

    # Forward pass to get the model output
    features = None
    gradients = None

    # Hook to capture gradients
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Hook to capture feature maps
    def forward_hook(module, input, output):
        nonlocal features
        features = output
        # Ensure gradient computation is retained
        features.retain_grad()

    # Register hooks to the last convolutional layer
    layer_name = "Mixed_7c.branch_pool.conv"  # Last InceptionV3 layer before FC
    layer = dict(model.named_modules())[layer_name]
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    target = output[0, class_idx]
    
    # Zero gradients & backward pass
    model.zero_grad()
    target.backward(retain_graph=True)

    # Get gradients and feature maps
    gradients = gradients.cpu().data.numpy()[0]
    features = features.cpu().data.numpy()[0]

    # Calculate Grad-CAM
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(features.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * features[i]

    # Apply ReLU
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (299, 299))  # Resize to match input size
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Convert to heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert tensor to image and overlay heatmap
    img = input_tensor.cpu().data.numpy()[0].transpose(1, 2, 0)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = np.uint8(img)

    # Overlay heatmap on original image
    overlayed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM output
    output_path = "gradcam_output.png"
    cv2.imwrite(output_path, overlayed_img)

    # Remove hooks to avoid memory leaks
    forward_handle.remove()
    backward_handle.remove()

    return output_path


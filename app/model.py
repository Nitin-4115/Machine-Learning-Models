import os
import sys
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in the path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from config import TRANSFORM
from utils import get_transform

# Grad-CAM global vars
gradients = None
activations = None

def load_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def register_hooks(model):
    def forward_hook(module, input, output):
        global activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0].detach()

    model.layer4.register_forward_hook(forward_hook)
    model.layer4.register_full_backward_hook(backward_hook)


def predict(image, model, class_names, top_k=3):
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)
    return [(class_names[idx], float(prob)) for idx, prob in zip(top_idxs, top_probs)]


def generate_gradcam(image: Image.Image, model, class_names):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # Store outputs
    activation = None
    gradient = None

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0].detach()

    # Hook into last conv layer
    try:
        target_layer = model.layer4[-1].conv3
    except AttributeError:
        raise RuntimeError("Failed to access model.layer4[-1].conv3. Check model architecture.")

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        class_idx = torch.argmax(output).item()
        score = output[0, class_idx]
        model.zero_grad()
        score.backward()

        # Ensure hooks captured values
        if activation is None or gradient is None:
            raise RuntimeError("Grad-CAM hook failed to capture activations or gradients.")

        act = activation.squeeze(0).cpu().numpy()
        grad = gradient.squeeze(0).cpu().numpy()

        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image.width, image.height))
        cam -= cam.min()
        cam /= cam.max()
        cam = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_np = np.array(image)
        overlayed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(overlayed)

    finally:
        forward_handle.remove()
        backward_handle.remove()

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import models
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
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    register_hooks(model)
    return model.to("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_gradcam(image, model, class_names=None):
    global activations, gradients

    device = next(model.parameters()).device
    input_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    # Reset activations and gradients
    activations = None
    gradients = None
    model.zero_grad()

    output = model(input_tensor)
    pred_class = output.argmax().item()
    class_label = class_names[pred_class] if class_names else f"Class {pred_class}"

    output[0, pred_class].backward()

    if activations is None or gradients is None:
        print("Grad-CAM warning: activations or gradients are None.")
        return image

    weights = torch.mean(gradients, dim=(2, 3))[0]
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)

    for i, w in enumerate(weights):
        cam += w * activations[0, i, :, :]

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.width, image.height))

    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(np.array(image.convert("RGB")), 0.5, heatmap, 0.5, 0)

    return Image.fromarray(overlayed)

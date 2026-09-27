"""
Grad-CAM Utility for DenseNet121
Generates visual heatmap overlays showing which regions the model focused on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """Grad-CAM for DenseNet121 — extracts gradients from the last conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for the given input."""
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global avg pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, probs.detach().cpu().numpy()[0]


def apply_heatmap(original_image, cam, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image.
    Returns: PIL Image with heatmap overlay.
    """
    # Resize CAM to match original image
    img_np = np.array(original_image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_np)

    return Image.fromarray(overlay)


def get_gradcam_for_densenet(model, input_tensor, original_image, target_class=None):
    """
    Complete Grad-CAM pipeline for DenseNet121.
    - model: The DenseNet121 model (with requires_grad enabled)
    - input_tensor: Preprocessed tensor [1, 3, 224, 224]
    - original_image: PIL Image (for overlay)
    - target_class: Class index (None = predicted class)
    Returns: (heatmap_image, predicted_class, probabilities)
    """
    # Enable gradients for Grad-CAM
    input_tensor.requires_grad_(True)

    # Get the last conv layer of DenseNet121
    # DenseNet121 features end with denseblock4 -> norm5
    if hasattr(model, 'base'):
        # DenseNetMC wrapper
        target_layer = model.base.features.denseblock4
    else:
        # Direct DenseNet
        target_layer = model.features.denseblock4

    gradcam = GradCAM(model, target_layer)
    cam, probs = gradcam.generate(input_tensor, target_class)

    # Create overlay image
    heatmap_image = apply_heatmap(original_image, cam, alpha=0.45)

    predicted_class = probs.argmax()

    return heatmap_image, int(predicted_class), probs

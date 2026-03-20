"""DINO-based image encoder helpers.

Provides a small utility to extract a fixed-size embedding from an
image crop using the DINOv2 vision transformer.
"""

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

# Choose device automatically (CUDA if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained DINOv2 model from torch.hub and set to eval mode.
# This returns a feature vector for an input image tensor.
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model = model.to(device)
model.eval()

# Standard image preprocessing used by many vision models. Resize to
# 224x224, convert to tensor and normalize with ImageNet means/stds.
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


@torch.no_grad()
def encode_bbox(frame_bgr, bbox_norm, out_dim=32):
    """Encode a normalized bounding box region into a DINO embedding.

    Args:
        frame_bgr: HxWx3 BGR NumPy array (as from OpenCV).
        bbox_norm: (x1, y1, x2, y2) normalized to [0,1] relative to frame.
        out_dim: desired output embedding length (truncates if needed).

    Returns:
        A Python list of length `out_dim` containing the normalized
        embedding. If the bbox is invalid or empty, returns zeros.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_norm

    # Convert normalized coords to integer pixel indices, clamped to frame
    x1 = max(0, min(w - 1, int(x1 * w)))
    y1 = max(0, min(h - 1, int(y1 * h)))
    x2 = max(0, min(w, int(x2 * w)))
    y2 = max(0, min(h, int(y2 * h)))

    # Guard against invalid boxes
    if x2 <= x1 or y2 <= y1:
        return [0.0] * out_dim

    # Crop the region from the frame; OpenCV images are BGR
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0] * out_dim

    # Convert to PIL Image with RGB channel order for transforms
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)

    # Preprocess and move to the chosen device
    x = transform(img).unsqueeze(0).to(device)

    # Forward pass through the model, detach and move to CPU
    feat = model(x).squeeze(0).detach().cpu()

    # L2-normalize for stable embeddings, avoid division by zero
    feat = feat / (feat.norm() + 1e-6)

    # Truncate or pad is not applied here; we simply take the first
    # `out_dim` elements of the feature vector (original code truncated)
    feat = feat[:out_dim]
    return feat.tolist()
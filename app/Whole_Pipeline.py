"""
=============================================================================
  Fundus Image Pipeline Demo (Whole_Pipeline.py)
  CF → FAF Translation → Disease Classification → Disease Segmentation
  Step-by-step interactive demo for project exhibition
=============================================================================
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
import io
import time
import cv2

# ── 1. MODEL ARCHITECTURES ──────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, features=48):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky")
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky")
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky")
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky")
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu")
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


# ── 2. GRAD-CAM IMPLEMENTATION ──────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Generates heatmaps showing which regions of the image contributed most
    to the classification decision. No additional training required.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep 
    Networks via Gradient-based Localization", ICCV 2017.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks on the target layer
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Class index to generate heatmap for. 
                         If None, uses the predicted class.
        
        Returns:
            cam: Normalized heatmap as numpy array (H, W), values in [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero all gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        # Global average pool the gradients → weights per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU — we only care about features that positively influence the class
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


def build_disease_colormap():
    """
    Build a custom colormap for disease probability visualization.
    Light yellow (low probability) → Orange → Red → Dark red (high probability).
    No black/dark regions — all values are visible and intuitive.
    """
    # 256-entry LUT: each entry is (B, G, R) for OpenCV
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.33:
            # Light yellow → Orange
            s = t / 0.33
            r = 255
            g = int(255 - s * 100)  # 255 → 155
            b = int(120 - s * 120)  # 120 → 0
        elif t < 0.66:
            # Orange → Red
            s = (t - 0.33) / 0.33
            r = 255
            g = int(155 - s * 125)  # 155 → 30
            b = 0
        else:
            # Red → Dark red
            s = (t - 0.66) / 0.34
            r = int(255 - s * 75)   # 255 → 180
            g = int(30 - s * 30)    # 30 → 0
            b = 0
        lut[i, 0] = [b, g, r]
    return lut

# Pre-build the colormap once at import time
DISEASE_COLORMAP_LUT = build_disease_colormap()


def apply_colormap(heatmap, colormap=None):
    """
    Convert a single-channel heatmap to a colored RGB image.
    Uses the custom disease colormap: light yellow → orange → red → dark red.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    # Apply the custom LUT
    colored = cv2.LUT(cv2.merge([heatmap_uint8, heatmap_uint8, heatmap_uint8]), DISEASE_COLORMAP_LUT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


# Bright blue contour color used everywhere (RGB 0-1 scale)
BLUE_CONTOUR = (0.0, 0.4, 1.0)
BLUE_CONTOUR_THICKNESS = 3


def create_overlay(image_np, heatmap, alpha=0.5):
    """
    Overlay a heatmap on an image using the disease colormap.
    
    Args:
        image_np: Original image as numpy array (H, W, 3), values in [0, 1]
        heatmap: Grad-CAM heatmap (H_cam, W_cam), values in [0, 1]
        alpha: Blending factor (0 = only image, 1 = only heatmap)
    
    Returns:
        overlaid: Blended image as numpy array (H, W, 3), values in [0, 1]
    """
    h, w = image_np.shape[:2]
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Apply custom disease colormap
    colored_heatmap = apply_colormap(heatmap_resized) / 255.0
    
    # Blend
    overlaid = (1 - alpha) * image_np + alpha * colored_heatmap
    return np.clip(overlaid, 0, 1)


def create_contour_overlay(image_np, heatmap, threshold=0.5, contour_color=(1.0, 0.2, 0.2), thickness=2):
    """
    Draw contours around thresholded heatmap regions on the image.
    
    Args:
        image_np: Original image as numpy array (H, W, 3), values in [0, 1]
        heatmap: Grad-CAM heatmap (H_cam, W_cam), values in [0, 1]
        threshold: Threshold value to binarize the heatmap
        contour_color: RGB color for contours (values 0-1)
        thickness: Contour line thickness in pixels
    
    Returns:
        contour_img: Image with contours drawn, numpy array (H, W, 3), values in [0, 1]
    """
    h, w = image_np.shape[:2]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Threshold to create binary mask
    binary_mask = (heatmap_resized >= threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the image
    contour_img = image_np.copy()
    if len(contour_img.shape) == 2 or contour_img.shape[2] == 1:
        contour_img = np.stack([contour_img.squeeze()] * 3, axis=-1)
    
    contour_img_uint8 = (contour_img * 255).astype(np.uint8)
    color_bgr = (int(contour_color[2] * 255), int(contour_color[1] * 255), int(contour_color[0] * 255))
    cv2.drawContours(contour_img_uint8, contours, -1, color_bgr, thickness)
    
    return contour_img_uint8.astype(np.float64) / 255.0


def create_combined_overlay(image_np, heatmap, threshold=0.4, alpha=0.55):
    """
    Create a combined visualization: disease heatmap ONLY inside contour + blue boundary.
    
    Inside the boundary:  dark red (high disease probability) → light yellow (lower probability)
    Outside the boundary: original FAF image for anatomical context
    Boundary:             bright blue contour
    """
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Step 1: Create binary mask of disease region
    binary_mask = (heatmap_resized >= threshold).astype(np.uint8)
    
    # Smooth the mask edges slightly for a cleaner look
    binary_mask_smooth = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 0)
    binary_mask_smooth = np.clip(binary_mask_smooth, 0, 1)
    mask_3ch = np.stack([binary_mask_smooth] * 3, axis=-1)
    
    # Step 2: Create heatmap overlay (custom disease colormap, only inside mask)
    colored_heatmap = apply_colormap(heatmap_resized) / 255.0
    heatmap_region = (1 - alpha) * image_np + alpha * colored_heatmap
    heatmap_region = np.clip(heatmap_region, 0, 1)
    
    # Step 3: Blend — heatmap inside mask, original image outside
    result = image_np * (1 - mask_3ch) + heatmap_region * mask_3ch
    result = np.clip(result, 0, 1)
    
    # Step 4: Draw bright blue contours on top
    contour_mask = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_uint8 = (result * 255).astype(np.uint8)
    # Blue contour in BGR: (255, 102, 0) → bright blue
    blue_bgr = (int(BLUE_CONTOUR[2] * 255), int(BLUE_CONTOUR[1] * 255), int(BLUE_CONTOUR[0] * 255))
    cv2.drawContours(result_uint8, contours, -1, blue_bgr, BLUE_CONTOUR_THICKNESS)
    
    return result_uint8.astype(np.float64) / 255.0


def compute_activation_ratio(heatmap, threshold=0.5):
    """
    Compute what percentage of the image has activation above the threshold.
    Normal images should have LOW activation ratio; disease images should have HIGHER.
    """
    binary = (heatmap >= threshold).astype(np.float32)
    return float(binary.sum() / binary.size) * 100


def generate_gradcam_for_faf(classifier, device, faf_np_array, target_class=None):
    """
    Full Grad-CAM pipeline: preprocess FAF → run Grad-CAM → return heatmap.
    
    Args:
        classifier: Loaded ResNet-18 model
        device: torch device
        faf_np_array: FAF image as numpy array (H, W, 3), values in [0, 1]
        target_class: Optional target class index
    
    Returns:
        heatmap: Grad-CAM heatmap as numpy array, values in [0, 1]
    """
    # Target the last convolutional block of ResNet-18 (layer4[-1])
    target_layer = classifier.layer4[-1]
    grad_cam = GradCAM(classifier, target_layer)
    
    # Preprocess (same as classification)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pil_img = Image.fromarray((faf_np_array * 255).astype(np.uint8))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Enable gradients for Grad-CAM (model must be in eval mode but with grad enabled)
    img_tensor.requires_grad_(True)
    
    # Generate heatmap
    heatmap = grad_cam.generate(img_tensor, target_class=target_class)
    
    return heatmap


# ── 3. CONFIGURATION ────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "cycleGAN_Weights_TrainOnly")
RESNET_WEIGHTS = os.path.join(BASE_DIR, "Codes", "Training", "Training_3_classes_TrainOnly", "resnet_faf_classifier.pth")
CLASS_NAMES = ["Choroiditis", "Normal", "Scar"]

# Find latest CycleGAN weight
CYCLEGAN_WEIGHT = None
if os.path.isdir(WEIGHTS_DIR):
    pth_files = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.startswith("G_AB") and f.endswith(".pth")])
    if pth_files:
        CYCLEGAN_WEIGHT = os.path.join(WEIGHTS_DIR, pth_files[-1])

# CF source directories for "safe" images
CF_DIR = os.path.join(BASE_DIR, "Dataset", "Final_Merged_CF_Renumbered")
TEST_LIST = os.path.join(BASE_DIR, "Dataset", "Splits", "classifier_cf_test.txt")
FAF_DATASET_DIR = os.path.join(BASE_DIR, "Dataset", "Generated_FAF_CNN_Dataset_3classes_TrainOnlyGAN")


# ── 4. HELPER FUNCTIONS ─────────────────────────────────────────────────────

@st.cache_resource
def load_cyclegan(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = UNetGenerator(in_channels=3, features=48).to(device)
    gen.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    gen.eval()
    return gen, device

@st.cache_resource
def load_resnet(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def translate_cf_to_faf(generator, device, pil_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(pil_image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        fake_faf = generator(img_tensor)
    output = fake_faf[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    return np.clip(output, 0, 1)

def classify_faf(classifier, device, faf_np_array):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pil_img = Image.fromarray((faf_np_array * 255).astype(np.uint8))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    return probs.cpu().numpy()

def get_cf_images_for_class(class_name):
    """Get the held-out test CF images for a specific disease class."""
    if not os.path.isfile(TEST_LIST):
        return []
    images = []
    with open(TEST_LIST) as handle:
        for line in handle:
            if not line.strip():
                continue
            folder, cls, name = line.strip().split("/")
            path = os.path.join(CF_DIR, cls, name)
            if cls == class_name and os.path.isfile(path):
                images.append(path)
    return sorted(images)


# ── 5. STREAMLIT UI ─────────────────────────────────────────────────────────

st.set_page_config(page_title="RetinalAI Pipeline Demo", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem 0;
    }
    .sub-header { color: #a0aec0; text-align: center; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .glass-card {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
        padding: 1.5rem; margin: 0.75rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(102,126,234,0.15); }
    .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 50%; font-weight: 700; font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea, #764ba2); color: white; margin-right: 0.5rem;
    }
    .step-title { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; display: inline; vertical-align: middle; }
    .image-frame {
        border: 2px solid rgba(102,126,234,0.3); border-radius: 12px; overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .image-label { text-align: center; padding: 0.6rem; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; }
    .cf-label { background: linear-gradient(90deg, rgba(56,189,248,0.2), rgba(59,130,246,0.2)); color: #93c5fd; }
    .faf-label { background: linear-gradient(90deg, rgba(52,211,153,0.2), rgba(16,185,129,0.2)); color: #6ee7b7; }
    .seg-label { background: linear-gradient(90deg, rgba(245,101,101,0.2), rgba(237,137,54,0.2)); color: #fc8181; }
    .fancy-divider { height: 2px; background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), transparent); margin: 1.5rem 0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #0f0c29); border-right: 1px solid rgba(102,126,234,0.2); }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2); color: white; border: none;
        border-radius: 8px; font-weight: 600; padding: 0.6rem 1.5rem; transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
    .confidence-bar-bg {
        background: rgba(255,255,255,0.08); border-radius: 8px; height: 28px;
        margin: 4px 0; overflow: hidden; position: relative;
    }
    .confidence-bar-fill { height: 100%; border-radius: 8px; transition: width 0.8s ease; }
    .confidence-text {
        position: absolute; top: 0; left: 10px; line-height: 28px;
        font-size: 0.82rem; font-weight: 600; color: white; text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }
    .prediction-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
        border: 2px solid rgba(102,126,234,0.4); border-radius: 16px; padding: 1.5rem;
        text-align: center; margin: 1rem 0;
    }
    .prediction-label { font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .warning-card {
        background: rgba(237,137,54,0.1); border: 1px solid rgba(237,137,54,0.3);
        border-radius: 12px; padding: 1rem; margin: 0.5rem 0; color: #f6ad55; font-size: 0.88rem;
    }
    .safe-badge {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600;
        background: rgba(72,187,120,0.2); color: #68d391; border: 1px solid rgba(72,187,120,0.4);
    }
    .seg-metric-card {
        background: rgba(245,101,101,0.08); border: 1px solid rgba(245,101,101,0.2);
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .seg-metric-value {
        font-size: 1.8rem; font-weight: 800;
        background: linear-gradient(90deg, #f56565, #ed8936);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .seg-metric-label { color: #a0aec0; font-size: 0.8rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <div style="font-size: 3rem;">🔬</div>
    <div style="font-size: 1.3rem; font-weight: 700;
                background: linear-gradient(90deg, #667eea, #764ba2);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        RetinalAI Pipeline
    </div>
    <div style="color: #718096; font-size: 0.75rem; margin-top: 0.3rem;">
        CF → FAF → Classification → Grad-CAM
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
            border: 1px solid rgba(102,126,234,0.3); border-radius: 12px; padding: 1rem; text-align: center;">
    <div style="color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Compute Device</div>
    <div style="font-size: 1.1rem; color: {'#68d391' if torch.cuda.is_available() else '#f6ad55'}; font-weight: 600; margin-top: 0.3rem;">
        {'🚀' if torch.cuda.is_available() else '🖥️'} {device_name}
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Data leakage info
    st.markdown("""
<div style="background: rgba(99,179,237,0.1); border: 1px solid rgba(99,179,237,0.3);
            border-radius: 12px; padding: 1rem; margin-top: 0.5rem;">
    <div style="color: #63b3ed; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">
        ℹ️ Held-Out Test Photos
    </div>
    <div style="color: #a0aec0; font-size: 0.78rem; line-height: 1.6;">
        Both models were retrained without <b>22 held-out patients</b>.
        The dataset choices in Step 1 are the <b>13 classifier photos</b> of those patients, so neither the CycleGAN nor the classifier saw them.
        <br><br>
        In testing, the classifier got <b>7 of these 13 right</b> (always answering Choroiditis gets 6).
        In this app it gets 8, because here the FAF skips the JPEG save the test images went through.
        Treat a single prediction as a demo, not a diagnosis.
        <br><br>
        An uploaded photo can come from anywhere, including the training set.
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Segmentation info
    st.markdown("""
<div style="background: rgba(245,101,101,0.08); border: 1px solid rgba(245,101,101,0.2);
            border-radius: 12px; padding: 1rem; margin-top: 0.5rem;">
    <div style="color: #fc8181; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">
        🔍 Heatmap Method
    </div>
    <div style="color: #a0aec0; font-size: 0.78rem; line-height: 1.6;">
        <b>Grad-CAM</b> (Gradient-weighted Class Activation Mapping) highlights the regions
        that most influenced the classifier's decision.
        <br><br>
        It uses <b>no additional training</b> — the heatmap is computed from the gradients of 
        the ResNet-18's last convolutional layer (<code>layer4</code>).
        <br><br>
        It shows <b>where the classifier looked</b>, not a segmentation of the disease.
        <br><br>
        <b>Reference:</b> Selvaraju et al., ICCV 2017
    </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN CONTENT ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">RetinalAI — Full Pipeline Demo</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-header">
    Upload a Colour Fundus image → Generate FAF → Classify Disease → Show Grad-CAM Heatmap<br>
    <span style="font-size: 0.85rem; color: #718096;">3-Class Model: Choroiditis · Normal · Scar &nbsp;|&nbsp; Grad-CAM Heatmap</span>
</div>
""", unsafe_allow_html=True)

# Check models exist
models_ok = True
if CYCLEGAN_WEIGHT is None:
    st.error("❌ No CycleGAN weights found in `cycleGAN_Weights_TrainOnly/`.")
    models_ok = False
if not os.path.isfile(RESNET_WEIGHTS):
    st.error(f"❌ ResNet weights not found at `{RESNET_WEIGHTS}`.")
    models_ok = False

if not models_ok:
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# STEP 1: Upload / Select CF Image
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top: 1rem;">
    <span class="step-badge">1</span>
    <span class="step-title">Select a Colour Fundus (CF) Image</span>
</div>
""", unsafe_allow_html=True)

input_method = st.radio(
    "Input method", ["📁 Upload Image", "🗂️ Choose a Held-Out Test Photo (Choroiditis / Normal / Scar)"],
    horizontal=True, label_visibility="collapsed"
)

input_image = None
source_class = None
input_key = None

if input_method == "📁 Upload Image":
    uploaded = st.file_uploader("Upload a CF Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded:
        input_image = Image.open(uploaded).convert('RGB')
        input_key = ("upload", uploaded.file_id)
else:
    sel_class = st.selectbox("Select disease class:", CLASS_NAMES)
    cf_images = get_cf_images_for_class(sel_class)
    if cf_images:
        sel_file = st.selectbox("Select image:", [os.path.basename(f) for f in cf_images])
        sel_path = os.path.join(CF_DIR, sel_class, sel_file)
        input_image = Image.open(sel_path).convert('RGB')
        source_class = sel_class
        input_key = ("dataset", sel_class, sel_file)
    else:
        st.warning(f"No CF images found for {sel_class}.")

if st.session_state.get('input_key') != input_key:
    st.session_state.input_key = input_key
    st.session_state.faf_result = None
    st.session_state.show_classify = False
    st.session_state.classification = None
    st.session_state.gradcam_result = None

if input_image is not None:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Show the input image
    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.markdown("""
<div class="image-frame">
    <div class="image-label cf-label">📷 Input Colour Fundus (CF)</div>
</div>
""", unsafe_allow_html=True)
        st.image(input_image, use_container_width=True)
        
    with col_info:
        weight_name = os.path.basename(CYCLEGAN_WEIGHT)
        
        # BULLETPROOF FIX: We build the string completely flush left, and conditionally add 
        # the ground truth ONLY if it exists, without introducing any empty newlines to the markdown parser.
        info_html = f"""
<div class="glass-card">
<div style="color: #93c5fd; font-weight: 700; margin-bottom: 0.8rem;">📋 Image Info</div>
<div style="color: #cbd5e0; font-size: 0.9rem;">
<div style="margin-bottom:0.3rem;"><span style="color:#a0aec0;">Resolution:</span> <b>{input_image.size[0]} × {input_image.size[1]}</b></div>
<div style="margin-bottom:0.3rem;"><span style="color:#a0aec0;">Mode:</span> <b>{input_image.mode}</b></div>
"""
        if source_class:
            info_html += f'<div style="margin-bottom:0.3rem;"><span style="color:#a0aec0;">Ground Truth:</span> <b>{source_class}</b></div>\n'
            
        info_html += f"""<div style="margin-bottom:0.3rem;"><span style="color:#a0aec0;">CycleGAN Weight:</span> <b>{weight_name}</b></div>
</div>
</div>
"""
        st.markdown(info_html.strip(), unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: Convert CF → FAF
    # ────────────────────────────────────────────────────────────────────────

    st.markdown("""
<div>
    <span class="step-badge">2</span>
    <span class="step-title">Convert CF → FAF (CycleGAN)</span>
</div>
""", unsafe_allow_html=True)

    if 'faf_result' not in st.session_state:
        st.session_state.faf_result = None
    if 'show_classify' not in st.session_state:
        st.session_state.show_classify = False

    st.write("") # Keep standard spacer here for generating button

    if st.button("🔄 Generate FAF Image", type="primary", key="gen_faf"):
        with st.spinner("🧠 Loading CycleGAN & translating CF → FAF..."):
            generator, device = load_cyclegan(CYCLEGAN_WEIGHT)
            start = time.time()
            faf_output = translate_cf_to_faf(generator, device, input_image)
            elapsed = time.time() - start
        st.session_state.faf_result = faf_output
        st.session_state.faf_time = elapsed
        st.session_state.show_classify = False
        st.session_state.classification = None
        st.session_state.gradcam_result = None

    if st.session_state.faf_result is not None:
        faf_output = st.session_state.faf_result
        elapsed = st.session_state.get('faf_time', 0)

        col_cf, col_arrow, col_faf = st.columns([5, 1, 5])
        with col_cf:
            st.markdown('<div class="image-frame"><div class="image-label cf-label">📷 Original CF</div></div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
        with col_arrow:
            st.markdown('<div style="display:flex;justify-content:center;align-items:center;height:100%;padding-top:6rem;"><div style="font-size:3rem;color:#667eea;">→</div></div>', unsafe_allow_html=True)
        with col_faf:
            st.markdown('<div class="image-frame"><div class="image-label faf-label">🖼️ Generated FAF</div></div>', unsafe_allow_html=True)
            st.image(faf_output, use_container_width=True, clamp=True)

        st.markdown(f"""
<div style="text-align:center; color:#a0aec0; font-size:0.85rem; margin-top:0.5rem;">
    Inference Time: <b>{elapsed:.3f}s</b> &nbsp;|&nbsp; Output: <b>256×256</b>
</div>
""", unsafe_allow_html=True)

        # Download FAF
        faf_pil = Image.fromarray((faf_output * 255).astype(np.uint8))
        buf = io.BytesIO()
        faf_pil.save(buf, format="PNG")
        st.download_button("📥 Download FAF", data=buf.getvalue(), file_name="generated_faf.png", mime="image/png")

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────────
        # STEP 3: Classify Disease
        # ────────────────────────────────────────────────────────────────────

        # FIX: Added a precise margin-bottom of 0.8rem here and removed st.write("") spacer 
        # so the button sits cleanly beneath the text without sticking to it or being pushed too far down.
        st.markdown("""
<div style="margin-bottom: 0.8rem;">
    <span class="step-badge">3</span>
    <span class="step-title">Classify Disease (ResNet-18)</span>
</div>
""", unsafe_allow_html=True)

        if st.button("🏥 Detect Disease", type="primary", key="classify"):
            classifier, cls_device = load_resnet(RESNET_WEIGHTS)
            with st.spinner("🔍 Running classification..."):
                probs = classify_faf(classifier, cls_device, faf_output)
            st.session_state.classification = probs
            st.session_state.show_classify = True
            st.session_state.gradcam_result = None

        if st.session_state.get('classification') is not None:
            probs = st.session_state.classification
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            pred_conf = probs[pred_idx] * 100

            # Prediction card
            emoji = "🟢" if pred_class == "Normal" else "🔴"
            st.markdown(f"""
<div class="prediction-card">
    <div style="font-size: 1rem; color: #a0aec0; margin-bottom: 0.5rem;">Predicted Classification</div>
    <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">{emoji}</div>
    <div class="prediction-label">{pred_class}</div>
    <div style="color: #cbd5e0; font-size: 1.1rem; margin-top: 0.5rem;">
        Confidence: <b>{pred_conf:.1f}%</b>
    </div>
</div>
""", unsafe_allow_html=True)

            # Confidence bars for all classes
            st.markdown("""
<div class="glass-card">
    <div style="color: #93c5fd; font-weight: 700; margin-bottom: 0.8rem;">📊 Confidence Scores</div>
""", unsafe_allow_html=True)

            colors = {"Choroiditis": "#f56565", "Normal": "#48bb78", "Scar": "#ed8936"}
            sorted_indices = np.argsort(probs)[::-1]

            for idx in sorted_indices:
                name = CLASS_NAMES[idx]
                conf = probs[idx] * 100
                color = colors.get(name, "#667eea")
                is_pred = " ★" if idx == pred_idx else ""
                st.markdown(f"""
<div class="confidence-bar-bg">
    <div class="confidence-bar-fill" style="width: {max(conf, 2):.1f}%; background: linear-gradient(90deg, {color}, {color}99);"></div>
    <div class="confidence-text">{name}{is_pred} — {conf:.1f}%</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Ground truth comparison if available
            if source_class:
                match = source_class == pred_class
                st.markdown(f"""
<div class="glass-card" style="border-color: {'rgba(72,187,120,0.4)' if match else 'rgba(245,101,101,0.4)'};">
    <div style="color: {'#68d391' if match else '#fc8181'}; font-weight: 700;">
        {'✅ Correct!' if match else '❌ Mismatch'} &nbsp;
        Ground Truth: <b>{source_class}</b> → Predicted: <b>{pred_class}</b>
    </div>
</div>
""", unsafe_allow_html=True)

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

            # ────────────────────────────────────────────────────────────────
            # STEP 4: Disease Region Segmentation (Grad-CAM)
            # ────────────────────────────────────────────────────────────────

            st.markdown("""
<div style="margin-bottom: 0.8rem;">
    <span class="step-badge">4</span>
    <span class="step-title">Where the Classifier Looked (Grad-CAM)</span>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div style="color: #a0aec0; font-size: 0.88rem; margin-bottom: 1rem; line-height: 1.6;">
    Grad-CAM highlights the <b>regions of the FAF image that most influenced the classification decision</b>.
    The left image shows the generated FAF; the right shows the Grad-CAM overlay. It shows where the classifier looked, not a segmentation of the disease.
</div>
""", unsafe_allow_html=True)

            if st.button("🔬 Show Grad-CAM Heatmap", type="primary", key="segment"):
                classifier, cls_device = load_resnet(RESNET_WEIGHTS)
                with st.spinner("🧬 Computing Grad-CAM heatmap..."):
                    start_seg = time.time()
                    heatmap = generate_gradcam_for_faf(classifier, cls_device, faf_output, target_class=pred_idx)
                    seg_time = time.time() - start_seg
                st.session_state.gradcam_result = heatmap
                st.session_state.seg_time = seg_time

            if st.session_state.get('gradcam_result') is not None:
                heatmap = st.session_state.gradcam_result
                seg_time = st.session_state.get('seg_time', 0)

                # Fixed optimal threshold (no slider — cleaner demo)
                threshold = 0.4

                # ── Create visualizations ──
                # Combined overlay: heatmap ONLY inside disease region + contours
                combined_overlay = create_combined_overlay(
                    faf_output, heatmap, threshold=threshold, alpha=0.55
                )

                # Activation ratio
                activation_pct = compute_activation_ratio(
                    cv2.resize(heatmap, (faf_output.shape[1], faf_output.shape[0])),
                    threshold=threshold
                )

                # ── Side-by-side display: Clean FAF | Segmented FAF ──
                st.markdown("""
<div style="color: #93c5fd; font-weight: 700; font-size: 1.1rem; margin: 1rem 0 0.5rem 0;">
    🔬 Grad-CAM Overlay
</div>
""", unsafe_allow_html=True)

                col_clean, col_seg = st.columns(2)
                with col_clean:
                    st.markdown('<div class="image-frame"><div class="image-label faf-label">🖼️ Generated FAF (Original)</div></div>', unsafe_allow_html=True)
                    st.image(faf_output, use_container_width=True, clamp=True)

                with col_seg:
                    st.markdown('<div class="image-frame"><div class="image-label seg-label">🔥 Grad-CAM Overlay</div></div>', unsafe_allow_html=True)
                    st.image(combined_overlay, use_container_width=True, clamp=True)

                # ── Metrics row ──
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f"""
<div class="seg-metric-card">
    <div class="seg-metric-value">{activation_pct:.1f}%</div>
    <div class="seg-metric-label">Activation Coverage</div>
</div>
""", unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
<div class="seg-metric-card">
    <div class="seg-metric-value">{seg_time*1000:.0f}ms</div>
    <div class="seg-metric-label">Grad-CAM Time</div>
</div>
""", unsafe_allow_html=True)
                with col_m3:
                    # Interpretation based on activation and class
                    if pred_class == "Normal":
                        interp_text = "Low" if activation_pct < 15 else "Moderate"
                        interp_color = "#48bb78" if activation_pct < 15 else "#f6ad55"
                    else:
                        interp_text = "High" if activation_pct > 10 else "Focal"
                        interp_color = "#f56565" if activation_pct > 10 else "#ed8936"
                    st.markdown(f"""
<div class="seg-metric-card">
    <div class="seg-metric-value" style="background: {interp_color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{interp_text}</div>
    <div class="seg-metric-label">Heatmap Spread</div>
</div>
""", unsafe_allow_html=True)

                # ── Additional view: Pure heatmap ──
                with st.expander("🎨 View Individual Visualizations", expanded=False):
                    col_heat, col_contour = st.columns(2)
                    
                    heatmap_overlay = create_overlay(faf_output, heatmap, alpha=0.5)
                    contour_overlay = create_contour_overlay(faf_output, heatmap, threshold=threshold, 
                                                            contour_color=BLUE_CONTOUR, thickness=BLUE_CONTOUR_THICKNESS)
                    
                    with col_heat:
                        st.markdown('<div class="image-label" style="color:#fc8181;">Heatmap Only</div>', unsafe_allow_html=True)
                        st.image(heatmap_overlay, use_container_width=True, clamp=True)
                    with col_contour:
                        st.markdown('<div class="image-label" style="color:#fc8181;">Contour Only</div>', unsafe_allow_html=True)
                        st.image(contour_overlay, use_container_width=True, clamp=True)

                # ── Download segmented image ──
                seg_pil = Image.fromarray((combined_overlay * 255).astype(np.uint8))
                seg_buf = io.BytesIO()
                seg_pil.save(seg_buf, format="PNG")
                st.download_button("📥 Download Grad-CAM Overlay", data=seg_buf.getvalue(), 
                                  file_name="gradcam_faf.png", mime="image/png", key="dl_seg")


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.75rem; padding: 1rem 0;">
    RetinalAI Pipeline Demo • CF → FAF → Classification → Grad-CAM • Built with Streamlit & PyTorch<br>
    <span style="color: #2d3748;">Heatmaps: Grad-CAM (Selvaraju et al., ICCV 2017) • No additional training required</span>
</div>
""", unsafe_allow_html=True)
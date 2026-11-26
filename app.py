"""
HeLa Cell Segmentation – Streamlit Demo
========================================
Weights are downloaded automatically from Hugging Face Hub on first run.
"""

import io
import os
import pathlib

import numpy as np
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

HF_REPO_ID  = "taijas469/hela-cell-segmentation"
WEIGHTS_FILE = "best_unet_model.pt"
LOCAL_WEIGHTS = pathlib.Path("best_unet_model.pt")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (512, 512)


# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD WEIGHTS FROM HUGGING FACE (only once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading model weights from Hugging Face…")
def load_model():
    if not LOCAL_WEIGHTS.exists():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILE)
        LOCAL_WEIGHTS.write_bytes(pathlib.Path(path).read_bytes())

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        decoder_attention_type="scse",
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(str(LOCAL_WEIGHTS), map_location=DEVICE)
    )
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE  (pure numpy / PIL – no cv2 dependency)
# ──────────────────────────────────────────────────────────────────────────────
def pil_resize(arr: np.ndarray, size: tuple, nearest: bool = False) -> np.ndarray:
    """Resize a 2-D numpy array via PIL (avoids cv2 dependency)."""
    pil = Image.fromarray(arr)
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return np.array(pil.resize(size, resample))


def run_inference(raw_bytes: bytes, threshold: float = 0.5):
    """
    Returns: (original_uint8_rgb, mask_uint8, overlay_uint8_rgb, prob_map)
    """
    # ── Load image ──────────────────────────────────────────────────────────
    pil_img = Image.open(io.BytesIO(raw_bytes))

    # Flatten to grayscale
    if pil_img.mode not in ("L", "I;16", "I"):
        pil_img = pil_img.convert("L")
    image = np.array(pil_img)

    # Normalise to float32 [0, 1]
    orig_dtype = image.dtype
    img_float  = image.astype(np.float32)
    max_val    = img_float.max()
    if max_val > 0:
        img_float /= max_val

    orig_h, orig_w = image.shape[:2]

    # Resize to model input
    img_resized = pil_resize(img_float, TARGET_SIZE, nearest=False)

    # Tensor [1, 1, H, W]
    tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logit = load_model()(tensor)
    prob  = torch.sigmoid(logit).squeeze().cpu().numpy()   # (512, 512)
    mask  = (prob >= threshold).astype(np.uint8) * 255

    # Upscale mask to original size
    mask_orig = pil_resize(mask, (orig_w, orig_h), nearest=True)

    # ── 8-bit display image ─────────────────────────────────────────────────
    if orig_dtype == np.uint16:
        display = ((image / image.max()) * 255).astype(np.uint8)
    else:
        display = np.clip(image, 0, 255).astype(np.uint8)
    display_rgb = np.stack([display, display, display], axis=-1)

    # ── Colour overlay ──────────────────────────────────────────────────────
    overlay      = display_rgb.copy()
    green_pixels = mask_orig > 0

    # Semi-transparent green fill
    overlay[green_pixels, 0] = (overlay[green_pixels, 0] * 0.55).astype(np.uint8)
    overlay[green_pixels, 1] = np.clip(
        overlay[green_pixels, 1] * 0.55 + 180 * 0.45, 0, 255
    ).astype(np.uint8)
    overlay[green_pixels, 2] = (overlay[green_pixels, 2] * 0.55).astype(np.uint8)

    # Red contour via edge detection on mask
    mask_pil     = Image.fromarray(mask_orig)
    # Erode then subtract to get boundary
    from PIL import ImageFilter
    eroded       = mask_pil.filter(ImageFilter.MinFilter(3))
    boundary     = np.array(mask_pil).astype(int) - np.array(eroded).astype(int)
    boundary     = (boundary > 0)
    overlay[boundary, 0] = 220
    overlay[boundary, 1] = 30
    overlay[boundary, 2] = 30

    return display_rgb, mask_orig, overlay, prob


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeLa Cell Segmentation",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align:center'>🔬 HeLa Cell Segmentation</h1>
    <p style='text-align:center; color:gray'>
        U-Net · ResNet-34 encoder · ComboLoss (BCE + Dice) · <b>95.9 % Dice Score</b>
    </p>
    <hr/>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Prediction threshold",
        min_value=0.1, max_value=0.95,
        value=0.5, step=0.05,
        help="Pixels with probability ≥ threshold are classified as cell.",
    )
    show_prob_map = st.checkbox("Show probability heat-map", value=False)
    st.markdown("---")
    st.markdown(
        f"**Device:** `{DEVICE}`  \n"
        f"**Model:** ResNet-34 U-Net  \n"
        f"**Input size:** 512 × 512  \n"
        f"**Loss:** BCE + Dice (ComboLoss)"
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Image is resized to 512×512  \n"
        "2. U-Net predicts per-pixel probability  \n"
        "3. Threshold applied to get binary mask  \n"
        "4. Overlay drawn on original image"
    )

# Upload
uploaded = st.file_uploader(
    "Upload a microscopy image (.tif / .tiff / .png)",
    type=["tif", "tiff", "png"],
)

if uploaded is None:
    # Sample output preview
    sample = pathlib.Path("outputs/sample_prediction_overlay.png")
    st.info("👆 Upload a HeLa cell microscopy image (.tif) to run segmentation.")
    if sample.exists():
        st.markdown("### 🖼️ Sample output from our training set")
        st.image(str(sample), caption="Green = predicted cells · Red = cell boundaries", use_container_width=True)
else:
    raw_bytes = uploaded.read()

    with st.spinner("Running segmentation…"):
        try:
            orig_rgb, mask, overlay, prob_map = run_inference(raw_bytes, threshold)
        except Exception as e:
            st.error(f"❌ Error during inference: {e}")
            st.stop()

    # Metrics row
    total_px = mask.size
    cell_px  = int((mask > 0).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Cell area", f"{cell_px / total_px * 100:.1f} %")
    m2.metric("Cell pixels", f"{cell_px:,}")
    m3.metric("Threshold", f"{threshold:.2f}")

    st.markdown("---")

    # Three-panel result
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📷 Original")
        st.image(orig_rgb, use_container_width=True)
    with c2:
        st.markdown("#### 🩹 Predicted mask")
        st.image(mask, clamp=True, use_container_width=True)
    with c3:
        st.markdown("#### 🎨 Overlay")
        st.image(overlay, use_container_width=True)
        st.caption("🟩 Green = predicted cells  |  🔴 Red = boundaries")

    # Probability heat-map
    if show_prob_map:
        st.markdown("---")
        st.markdown("#### 🌡️ Raw probability map")
        # Jet colormap without cv2
        prob_norm = (prob_map * 255).astype(np.uint8)
        prob_pil  = Image.fromarray(prob_norm, mode="L").convert("RGB")
        st.image(np.array(prob_pil), use_container_width=True,
                 caption="Brighter = higher cell probability")

    # Download
    st.markdown("---")
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    st.download_button(
        "⬇️ Download overlay as PNG",
        data=buf.getvalue(),
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}_segmented.png",
        mime="image/png",
    )

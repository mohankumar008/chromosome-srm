"""
app.py
------
Step 11 of the pipeline: Streamlit demo.

    Upload chromosome image
            |
            v
        [Enhance]
            |
            v
       Trained EDSR (falls back to SRCNN, then Bicubic, if checkpoints
       are missing, so the demo always runs even before training)
            |
            v
    Display: Original | Super-Resolved
    Metrics: PSNR / SSIM (only if the user also has a ground-truth HR
             image to compare against) + processing time

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.srcnn import SRCNN
from models.edsr import EDSR
from evaluation.metrics import compute_psnr, compute_ssim

st.set_page_config(page_title="Chromosome Super-Resolution", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EDSR_CKPT = "checkpoints/edsr_best.pth"
SRCNN_CKPT = "checkpoints/srcnn_best.pth"
SCALE = 4
EDSR_NUM_FEATURES = 64
EDSR_NUM_RES_BLOCKS = 12  # must match the --num_res_blocks used in training/train.py


@st.cache_resource
def load_models():
    edsr = srcnn = None
    if os.path.exists(EDSR_CKPT):
        edsr = EDSR(num_channels=1, num_features=EDSR_NUM_FEATURES, num_res_blocks=EDSR_NUM_RES_BLOCKS, scale=SCALE).to(DEVICE)
        edsr.load_state_dict(torch.load(EDSR_CKPT, map_location=DEVICE))
        edsr.eval()
    if os.path.exists(SRCNN_CKPT):
        srcnn = SRCNN(num_channels=1).to(DEVICE)
        srcnn.load_state_dict(torch.load(SRCNN_CKPT, map_location=DEVICE))
        srcnn.eval()
    return edsr, srcnn


def read_uploaded_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("L")
    return np.array(image)


def run_bicubic(lr01: np.ndarray, scale: int) -> np.ndarray:
    h, w = lr01.shape
    return np.clip(cv2.resize(lr01, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC), 0, 1)


def run_edsr(model, lr01: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(lr01.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = model(t)
        return out.squeeze().clamp(0, 1).cpu().numpy()


def run_srcnn(model, lr01_upsampled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(lr01_upsampled.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = model(t)
        return out.squeeze().clamp(0, 1).cpu().numpy()


def to_png_bytes(img01: np.ndarray) -> bytes:
    img_u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img_u8)
    return buf.tobytes()


def main():
    st.title("🧬 Chromosome Image Super-Resolution")
    st.caption(
        "Deep-learning-based reconstruction of high-resolution chromosome imagery "
        "from medium-resolution microscopy input — with quantitative fidelity checks, "
        "not just visual sharpening."
    )

    edsr_model, srcnn_model = load_models()

    with st.sidebar:
        st.header("Settings")
        method = st.selectbox(
            "Super-resolution method",
            options=[m for m, avail in [
                ("EDSR (recommended)", edsr_model is not None),
                ("SRCNN", srcnn_model is not None),
                ("Bicubic (baseline)", True),
            ] if avail],
        )
        st.markdown("---")
        st.subheader("Optional: ground truth")
        gt_file = st.file_uploader(
            "Upload the matching high-resolution image to compute PSNR/SSIM",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="gt",
        )
        if edsr_model is None and srcnn_model is None:
            st.warning(
                "No trained checkpoints found in checkpoints/. "
                "Only the Bicubic baseline is available until you train a model."
            )

    uploaded_file = st.file_uploader(
        "Upload a medium/low-resolution chromosome image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    lr_u8 = read_uploaded_image(uploaded_file)
    lr01 = lr_u8.astype(np.float32) / 255.0

    if st.button("🔬 SUPER RESOLVE", type="primary"):
        start = time.time()

        if method.startswith("EDSR") and edsr_model is not None:
            sr01 = run_edsr(edsr_model, lr01)
        elif method.startswith("SRCNN") and srcnn_model is not None:
            up = run_bicubic(lr01, SCALE)
            sr01 = run_srcnn(srcnn_model, up)
        else:
            sr01 = run_bicubic(lr01, SCALE)

        elapsed = time.time() - start

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original (input)")
            st.image(lr_u8, clamp=True, use_container_width=True)
            st.caption(f"Size: {lr_u8.shape[1]} x {lr_u8.shape[0]}")
        with col2:
            st.subheader(f"Super-Resolved ({method})")
            st.image((sr01 * 255).astype(np.uint8), clamp=True, use_container_width=True)
            st.caption(f"Size: {sr01.shape[1]} x {sr01.shape[0]}")

        st.download_button(
            "Download super-resolved image",
            data=to_png_bytes(sr01),
            file_name="super_resolved.png",
            mime="image/png",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Processing time", f"{elapsed*1000:.0f} ms")

        if gt_file is not None:
            gt_u8 = read_uploaded_image(gt_file)
            gt01 = gt_u8.astype(np.float32) / 255.0
            if gt01.shape != sr01.shape:
                gt01_resized = cv2.resize(gt01, (sr01.shape[1], sr01.shape[0]))
            else:
                gt01_resized = gt01
            psnr_val = compute_psnr(gt01_resized, sr01)
            ssim_val = compute_ssim(gt01_resized, sr01)
            m2.metric("PSNR", f"{psnr_val:.2f} dB")
            m3.metric("SSIM", f"{ssim_val:.4f}")
        else:
            m2.metric("PSNR", "—")
            m3.metric("SSIM", "—")
            st.caption("Upload a ground-truth HR image in the sidebar to compute PSNR/SSIM.")

        st.markdown("---")
        st.warning(
            "⚠️ **Research note:** super-resolution models can enhance detail but may also "
            "hallucinate plausible-looking structures that aren't in the original sample. "
            "This tool is a research demonstration, not a diagnostic instrument — outputs "
            "should not be used for clinical interpretation of chromosome structure."
        )


if __name__ == "__main__":
    main()

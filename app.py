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
import zipfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.srcnn import SRCNN
from models.edsr import EDSR
from evaluation.metrics import compute_psnr, compute_ssim
from preprocessing.segment_chromosomes import segment_chromosomes

st.set_page_config(page_title="Chromosome Super-Resolution", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EDSR_CKPT = "checkpoints/edsr_best.pth"
SRCNN_CKPT = "checkpoints/srcnn_best.pth"
SCALE = 4
EDSR_NUM_FEATURES = 64
EDSR_NUM_RES_BLOCKS = 12  # must match the --num_res_blocks used in training/train.py

# Safety cap on the LARGEST side of any uploaded input image (before the scale
# factor is applied). The models were trained on single-chromosome crops up
# to 256x256. Full karyotype spreads (many chromosomes, ~900x1000px) are far
# outside that training distribution AND can exhaust memory when run through
# EDSR (12 residual blocks x 64 channels x a 4x upsample), especially on
# small cloud instances. Rather than crash, we downscale anything larger than
# this cap and warn the user, so the app always returns *something* safely.
MAX_INPUT_DIM = 256


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


def enhance_u8(lr_u8: np.ndarray, method: str, edsr_model, srcnn_model) -> np.ndarray:
    """Run the selected method on a single grayscale uint8 image, returning a
    float32 [0,1] output. Shared by both the single-image and the batch
    full-spread-segmentation modes so the two stay behaviourally identical."""
    lr01 = lr_u8.astype(np.float32) / 255.0
    if method.startswith("EDSR") and edsr_model is not None:
        return run_edsr(edsr_model, lr01)
    elif method.startswith("SRCNN") and srcnn_model is not None:
        up = run_bicubic(lr01, SCALE)
        return run_srcnn(srcnn_model, up)
    else:
        return run_bicubic(lr01, SCALE)


def render_single_mode(edsr_model, srcnn_model):
    with st.sidebar:
        st.header("Settings")
        method = st.selectbox(
            "Super-resolution method",
            options=[m for m, avail in [
                ("EDSR (recommended)", edsr_model is not None),
                ("SRCNN", srcnn_model is not None),
                ("Bicubic (baseline)", True),
            ] if avail],
            key="single_method",
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
        key="single_upload",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    lr_u8 = read_uploaded_image(uploaded_file)

    # Safety guard: downscale oversized inputs (e.g. a full multi-chromosome
    # karyotype spread) instead of letting the model run out of memory.
    h, w = lr_u8.shape
    longest_side = max(h, w)
    if longest_side > MAX_INPUT_DIM:
        resize_ratio = MAX_INPUT_DIM / longest_side
        new_w, new_h = max(1, int(w * resize_ratio)), max(1, int(h * resize_ratio))
        lr_u8 = cv2.resize(lr_u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        st.info(
            f"ℹ️ Input was {w}x{h}, larger than this demo's {MAX_INPUT_DIM}px "
            f"safety limit, so it was automatically downscaled to {new_w}x{new_h} "
            "before enhancement. This model was trained on single isolated "
            "chromosome crops, not full multi-chromosome spreads — for a "
            "meaningful result, use the 'Full Spread' tab instead to auto-detect "
            "and separate each chromosome first."
        )

    if st.button("🔬 SUPER RESOLVE", type="primary"):
        start = time.time()
        try:
            sr01 = enhance_u8(lr_u8, method, edsr_model, srcnn_model)
        except RuntimeError as e:
            st.error(
                "⚠️ Ran out of memory or hit a runtime error processing this "
                "image. Try a smaller image, or a single cropped chromosome "
                "rather than a full spread.\n\n"
                f"Technical detail: {e}"
            )
            return
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


def render_full_spread_mode(edsr_model, srcnn_model):
    st.markdown(
        "Upload a **full karyotype spread** (many chromosomes in one microscopy "
        "image). This automatically detects and separates each individual "
        "chromosome using classical image processing (thresholding + watershed "
        "segmentation), then runs super-resolution on each crop independently."
    )
    st.info(
        "ℹ️ **Known limitation:** chromosomes that heavily overlap or cross each "
        "other in a dense cluster may be detected as a single merged region rather "
        "than perfectly separated. This is a well-known hard problem in "
        "cytogenetics — even trained lab technicians separate dense overlaps "
        "manually. Isolated and lightly-touching chromosomes separate reliably."
    )

    method = st.selectbox(
        "Super-resolution method",
        options=[m for m, avail in [
            ("EDSR (recommended)", edsr_model is not None),
            ("SRCNN", srcnn_model is not None),
            ("Bicubic (baseline)", True),
        ] if avail],
        key="spread_method",
    )

    spread_file = st.file_uploader(
        "Upload a full karyotype spread image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key="spread_upload",
    )

    if spread_file is None:
        st.info("Upload a full spread image to begin.")
        return

    spread_u8 = read_uploaded_image(spread_file)

    if st.button("🧬 DETECT & SEPARATE CHROMOSOMES", type="primary"):
        with st.spinner("Segmenting chromosomes..."):
            try:
                crops, boxes, overview_bgr = segment_chromosomes(spread_u8)
            except Exception as e:
                st.error(f"⚠️ Segmentation failed: {e}")
                return

        if len(crops) == 0:
            st.warning("No chromosomes detected. Try a different image.")
            return

        st.success(f"Detected {len(crops)} chromosomes.")
        overview_rgb = cv2.cvtColor(overview_bgr, cv2.COLOR_BGR2RGB)
        st.image(overview_rgb, caption="Detected chromosomes (numbered)", use_container_width=True)

        # Cache results in session_state so the "Enhance All" button below
        # doesn't require re-running segmentation on every rerun.
        st.session_state["seg_crops"] = crops
        st.session_state["seg_method"] = method

    if "seg_crops" in st.session_state:
        crops = st.session_state["seg_crops"]
        method = st.session_state.get("seg_method", method)

        if st.button(f"✨ Enhance all {len(crops)} detected chromosomes with {method}"):
            enhanced_list = []
            progress = st.progress(0.0, text="Enhancing chromosomes...")
            for i, crop in enumerate(crops):
                try:
                    sr01 = enhance_u8(crop, method, edsr_model, srcnn_model)
                    enhanced_list.append(sr01)
                except RuntimeError:
                    enhanced_list.append(None)  # skip a crop that fails, don't kill the whole batch
                progress.progress((i + 1) / len(crops), text=f"Enhancing chromosomes... {i+1}/{len(crops)}")
            progress.empty()

            st.session_state["seg_enhanced"] = enhanced_list

        if "seg_enhanced" in st.session_state:
            enhanced_list = st.session_state["seg_enhanced"]
            ok_count = sum(1 for e in enhanced_list if e is not None)
            st.success(f"Enhanced {ok_count} / {len(crops)} chromosomes.")

            # Grid display, a handful of columns at a time
            cols_per_row = 6
            for row_start in range(0, len(crops), cols_per_row):
                row_items = list(enumerate(crops))[row_start:row_start + cols_per_row]
                cols = st.columns(len(row_items))
                for col, (i, crop) in zip(cols, row_items):
                    sr01 = enhanced_list[i]
                    with col:
                        if sr01 is not None:
                            st.image((sr01 * 255).astype(np.uint8), clamp=True, use_container_width=True)
                            st.caption(f"#{i+1}")
                        else:
                            st.caption(f"#{i+1} (failed)")

            # Zip download of all enhanced crops
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, sr01 in enumerate(enhanced_list, start=1):
                    if sr01 is not None:
                        zf.writestr(f"chromosome_{i:03d}_enhanced.png", to_png_bytes(sr01))
            st.download_button(
                "Download all enhanced chromosomes (ZIP)",
                data=zip_buf.getvalue(),
                file_name="enhanced_chromosomes.zip",
                mime="application/zip",
            )


def main():
    st.title("🧬 Chromosome Image Super-Resolution")
    st.caption(
        "Deep-learning-based reconstruction of high-resolution chromosome imagery "
        "from medium-resolution microscopy input — with quantitative fidelity checks, "
        "not just visual sharpening."
    )

    edsr_model, srcnn_model = load_models()

    tab1, tab2 = st.tabs(["Single Chromosome", "Full Spread → Auto-Separate + Enhance"])
    with tab1:
        render_single_mode(edsr_model, srcnn_model)
    with tab2:
        render_full_spread_mode(edsr_model, srcnn_model)

    st.markdown("---")
    st.warning(
        "⚠️ **Research note:** super-resolution models can enhance detail but may also "
        "hallucinate plausible-looking structures that aren't in the original sample. "
        "This tool is a research demonstration, not a diagnostic instrument — outputs "
        "should not be used for clinical interpretation of chromosome structure."
    )


if __name__ == "__main__":
    main()

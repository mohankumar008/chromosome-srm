"""
Automatic chromosome segmentation from a full karyotype spread image.

Given a raw microscopy image containing many chromosomes scattered on a
plain background (and possibly a round interphase nucleus), this module
detects and crops out each individual chromosome, so each crop can then be
fed through the super-resolution models (SRCNN/EDSR) independently.

Approach (classical image processing, no training required):
  1. Otsu thresholding (inverted) to separate dark chromosomes from the
     light background.
  2. Morphological open+close to remove speckle noise and close small gaps.
  3. Contour-based filtering to remove non-chromosome blobs:
       - too small  -> dust/noise
       - too large  -> almost certainly not a single chromosome
       - high "solidity" (contour area / convex-hull area) -> round blobs
         such as an interphase nucleus, which is solid and round, unlike
         a chromosome, which is bent/elongated and has low solidity
  4. Watershed segmentation (on the distance transform) to split
     TOUCHING chromosomes that would otherwise merge into one connected
     component. A minimum peak-distance and a post-merge step prevent a
     single bent chromosome from being incorrectly split at its own elbow.

Known limitation (state this honestly in the report): chromosomes that
heavily overlap/cross each other in a dense cluster may still be grouped
into one region rather than perfectly separated. This is a well-known hard
problem in cytogenetics (chromosome "declumping") and is not something a
purely classical approach fully solves -- even trained lab technicians
separate dense overlaps manually. A learned instance-segmentation model
(e.g. Mask R-CNN trained on the annotation masks) would be the natural next
step to improve this further and is a reasonable "future work" item.

Usage as a script:
    python preprocessing/segment_chromosomes.py --image path/to/spread.jpg --out_dir outputs/segmented

Usage as a module (e.g. from app.py):
    from preprocessing.segment_chromosomes import segment_chromosomes
    crops, boxes, overview_bgr = segment_chromosomes(gray_image_np_array)
"""

import argparse
import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def segment_chromosomes(
    img_gray,
    min_area=150,
    max_area_ratio=0.04,
    solidity_thresh=0.85,
    min_distance=28,
    min_fragment_area=250,
    pad_ratio=0.15,
):
    """
    Segment individual chromosomes out of a full karyotype spread.

    Args:
        img_gray: 2D numpy array (grayscale image), uint8.
        min_area: contours smaller than this (pixels) are treated as noise.
        max_area_ratio: contours larger than this fraction of the total
            image area are rejected (too big to be a single chromosome).
        solidity_thresh: contours with (area / convex_hull_area) above this
            are rejected as round blobs (e.g. an interphase nucleus).
        min_distance: minimum distance (px) between watershed peaks -- the
            main knob to prevent over-splitting a single bent chromosome.
        min_fragment_area: watershed regions smaller than this are merged
            into their largest touching neighbour rather than kept as a
            separate (likely spurious) detection.
        pad_ratio: extra margin added around each detected bounding box
            before cropping, as a fraction of the box's largest side.

    Returns:
        crops: list of 2D numpy arrays (grayscale crops), one per detected
            chromosome.
        boxes: list of (x0, y0, x1, y1) tuples, the crop coordinates in the
            original image, in the same order as `crops`.
        overview_bgr: a BGR copy of the input image with every detected box
            drawn and numbered, useful for visual sanity-checking.
    """
    H, W = img_gray.shape
    total_area = H * W

    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Remove non-chromosome blobs (noise, nucleus) before watershed splitting
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > total_area * max_area_ratio:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity > solidity_thresh:
            continue  # round blob (nucleus/debris)
        cv2.drawContours(clean_mask, [c], -1, 255, -1)

    # Watershed on the distance transform splits touching chromosomes
    distance = ndi.distance_transform_edt(clean_mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=clean_mask)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    labels_ws = watershed(-distance, markers, mask=clean_mask)

    # Merge tiny fragments (usually an over-split sliver) into their
    # largest touching neighbour instead of keeping them as junk detections
    unique_labels = [l for l in np.unique(labels_ws) if l != 0]
    areas = {l: int((labels_ws == l).sum()) for l in unique_labels}
    dilate_kernel = np.ones((5, 5), np.uint8)
    for l in unique_labels:
        if areas[l] >= min_fragment_area:
            continue
        comp = (labels_ws == l).astype(np.uint8)
        dilated = cv2.dilate(comp, dilate_kernel, iterations=2)
        neighbor_labels = np.unique(labels_ws[(dilated > 0) & (labels_ws != l) & (labels_ws != 0)])
        if len(neighbor_labels) > 0:
            biggest_neighbor = max(neighbor_labels, key=lambda x: areas.get(x, 0))
            labels_ws[labels_ws == l] = biggest_neighbor
            areas[biggest_neighbor] = areas.get(biggest_neighbor, 0) + areas[l]

    overview_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    crops, boxes = [], []
    idx = 0
    for label in np.unique(labels_ws):
        if label == 0:
            continue
        comp_mask = (labels_ws == label).astype(np.uint8)
        area = int(comp_mask.sum())
        if area < min_fragment_area:
            continue
        ys, xs = np.where(comp_mask)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)

        idx += 1
        pad = int(pad_ratio * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crops.append(img_gray[y0:y1, x0:x1])
        boxes.append((x0, y0, x1, y1))
        cv2.rectangle(overview_bgr, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(overview_bgr, str(idx), (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return crops, boxes, overview_bgr


def main():
    ap = argparse.ArgumentParser(description="Segment individual chromosomes from a full karyotype spread image")
    ap.add_argument("--image", required=True, help="Path to a full karyotype spread image (jpg/png)")
    ap.add_argument("--out_dir", default="outputs/segmented", help="Where to save the individual crops and overview")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    crops, boxes, overview_bgr = segment_chromosomes(img_gray)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    overview_path = os.path.join(args.out_dir, f"{stem}_overview.png")
    cv2.imwrite(overview_path, overview_bgr)

    for i, crop in enumerate(crops, start=1):
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_chr{i:03d}.png"), crop)

    print(f"Detected {len(crops)} chromosomes in {args.image}")
    print(f"Overview image: {overview_path}")
    print(f"Individual crops saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

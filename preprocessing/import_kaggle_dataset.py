"""
Import the aliabedimadiseh/chromosome-image-dataset-karyotype Kaggle dataset
into this project's dataset/{train,val,test}/HR/ layout.

The Kaggle dataset ships full karyotype spread images (many chromosomes per
image) plus one Pascal-VOC-style XML annotation file per image, each
containing a <bndbox> (xmin, ymin, xmax, ymax) for every individual
chromosome. This script:

  1. Parses every XML annotation
  2. Crops out each individual chromosome bounding box from its source JPEG
     (with a small padding margin so the chromosome isn't cut off at the edge)
  3. Pads each crop to a square (so resizing doesn't distort the chromosome's
     proportions) and resizes it to a fixed HR size
  4. Converts to grayscale (matches this project's models, which expect
     single-channel input)
  5. Splits crops into train/val/test by SOURCE IMAGE (not by individual
     chromosome) so all chromosomes from one karyotype spread land in the
     same split -- this avoids leaking near-duplicate information between
     splits
  6. Writes the results into dataset/train/HR, dataset/val/HR, dataset/test/HR

After running this, regenerate the LR pairs with create_lr_images.py exactly
as you did for the synthetic placeholder data -- no other code changes needed.

Usage (Windows example - run from inside the chromosome-srm project folder):
    python preprocessing/import_kaggle_dataset.py --images_dir "PASTE_JEPG_FOLDER_PATH_HERE" --annotations_dir "PASTE_ANNOTATIONS_FOLDER_PATH_HERE" --out_dir dataset --hr_size 256 --max_source_images 300
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def parse_annotation(xml_path):
    """Return (image_filename, img_w, img_h, [ (xmin,ymin,xmax,ymax,name), ... ])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.findtext("xmin")))
        ymin = int(float(bnd.findtext("ymin")))
        xmax = int(float(bnd.findtext("xmax")))
        ymax = int(float(bnd.findtext("ymax")))
        boxes.append((xmin, ymin, xmax, ymax, name))
    return filename, img_w, img_h, boxes


def crop_square_padded(img, xmin, ymin, xmax, ymax, pad_ratio=0.15):
    """Crop a box out of img, pad it to a square, with a margin around the box."""
    h, w = img.shape[:2]
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Expand the box by pad_ratio on each side
    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(w, xmax + pad_x)
    ymax = min(h, ymax + pad_y)

    crop = img[ymin:ymax, xmin:xmax]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    # Pad to square (centered) using edge-replication so we don't introduce
    # hard black borders that the model could learn as a spurious cue
    side = max(ch, cw)
    top = (side - ch) // 2
    bottom = side - ch - top
    left = (side - cw) // 2
    right = side - cw - left
    square = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return square


def main():
    ap = argparse.ArgumentParser(description="Import Kaggle karyotype dataset -> dataset/{train,val,test}/HR")
    ap.add_argument("--images_dir", required=True, help="Path to the JEPG folder of full karyotype spreads")
    ap.add_argument("--annotations_dir", required=True, help="Path to the annotations folder of XML files")
    ap.add_argument("--out_dir", default="dataset", help="Project dataset root (contains train/val/test)")
    ap.add_argument("--hr_size", type=int, default=256, help="Output size for each cropped chromosome (square)")
    ap.add_argument("--pad_ratio", type=float, default=0.15, help="Padding added around each bounding box")
    ap.add_argument("--min_box_px", type=int, default=20, help="Skip boxes smaller than this on either side (likely annotation noise)")
    ap.add_argument("--max_source_images", type=int, default=0, help="Optional cap on number of source spreads to process (0 = all). Useful for a quick first run.")
    ap.add_argument("--max_crops_per_image", type=int, default=0, help="Optional cap on chromosomes kept per source image (0 = all)")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    xml_files = sorted(f for f in os.listdir(args.annotations_dir) if f.lower().endswith(".xml"))
    if args.max_source_images > 0:
        random.shuffle(xml_files)
        xml_files = xml_files[: args.max_source_images]

    print(f"Found {len(xml_files)} annotation files to process.")

    # Split by SOURCE IMAGE so chromosomes from one spread stay together
    random.shuffle(xml_files)
    n = len(xml_files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train:n_train + n_val],
        "test": xml_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        out_hr_dir = os.path.join(args.out_dir, split_name, "HR")
        os.makedirs(out_hr_dir, exist_ok=True)

        crop_count = 0
        source_count = 0
        for xml_name in files:
            xml_path = os.path.join(args.annotations_dir, xml_name)
            try:
                filename, img_w, img_h, boxes = parse_annotation(xml_path)
            except Exception as e:
                print(f"  [skip] failed to parse {xml_name}: {e}")
                continue

            img_path = os.path.join(args.images_dir, filename)
            if not os.path.exists(img_path):
                # Some datasets have mismatched extensions/casing; try a couple of fallbacks
                stem = os.path.splitext(xml_name)[0]
                for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                    alt = os.path.join(args.images_dir, stem + ext)
                    if os.path.exists(alt):
                        img_path = alt
                        break
                else:
                    print(f"  [skip] image not found for {xml_name} (expected {filename})")
                    continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [skip] failed to read image {img_path}")
                continue

            source_count += 1
            kept_this_image = 0
            for (xmin, ymin, xmax, ymax, name) in boxes:
                if (xmax - xmin) < args.min_box_px or (ymax - ymin) < args.min_box_px:
                    continue
                square = crop_square_padded(img, xmin, ymin, xmax, ymax, args.pad_ratio)
                if square is None:
                    continue

                gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (args.hr_size, args.hr_size), interpolation=cv2.INTER_CUBIC)

                stem = os.path.splitext(filename)[0]
                out_name = f"{stem}_{name}_{crop_count:06d}.png"
                cv2.imwrite(os.path.join(out_hr_dir, out_name), resized)
                crop_count += 1
                kept_this_image += 1

                if args.max_crops_per_image and kept_this_image >= args.max_crops_per_image:
                    break

        print(f"[{split_name}] {source_count} source images -> {crop_count} cropped chromosome images -> {out_hr_dir}")

    print("\nDone. Next step: generate LR pairs for each split, e.g.")
    print('  python preprocessing/create_lr_images.py --hr_dir dataset/train/HR --lr_dir dataset/train/LR --scale 4 --blur --noise')
    print('  python preprocessing/create_lr_images.py --hr_dir dataset/val/HR   --lr_dir dataset/val/LR   --scale 4 --blur --noise')
    print('  python preprocessing/create_lr_images.py --hr_dir dataset/test/HR  --lr_dir dataset/test/LR  --scale 4 --blur --noise')


if __name__ == "__main__":
    main()
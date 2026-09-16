# Chromosome Image Super-Resolution — Final-Year Project

**Research question:** Can deep learning reconstruct high-resolution
chromosome imagery from medium-resolution microscopy data while
maintaining structural fidelity — and how does this compare to
conventional interpolation?

This repository is a full research + demo pipeline, not just an
"upscale a picture" script:

```
HR chromosome images → generate LR pairs → preprocessing (patches, norm)
    → train SRCNN / EDSR → evaluate (PSNR/SSIM, Bicubic vs SRCNN vs EDSR)
    → visual comparison → Streamlit demo
```

The code has been smoke-tested end-to-end on a small synthetic dataset
(included under `dataset/`, `checkpoints/`, `outputs/`) so you can confirm
everything runs before plugging in real chromosome images and training for
real. **Those synthetic results are not meaningful science — they exist
only to prove the pipeline works.** Replace the dataset with real,
licensed chromosome microscopy images before drawing any conclusions.

## 1. Project structure

```
chromosome-srm/
├── dataset/
│   ├── train/{HR,LR}
│   ├── val/{HR,LR}
│   └── test/{HR,LR}
├── models/
│   ├── srcnn.py                  # baseline deep model
│   └── edsr.py                   # main model (residual CNN + PixelShuffle)
├── preprocessing/
│   ├── create_lr_images.py       # HR -> LR degradation pipeline
│   └── generate_synthetic_dataset.py  # pipeline smoke-test only
├── training/
│   ├── dataset.py                # PyTorch Dataset + patch cropping
│   └── train.py                  # training loop, logs loss/PSNR/SSIM
├── evaluation/
│   ├── metrics.py                # PSNR / SSIM
│   ├── bicubic_baseline.py       # classical baseline (run this FIRST)
│   └── evaluate.py               # Bicubic vs SRCNN vs EDSR comparison
├── checkpoints/                  # saved .pth model weights
├── outputs/
│   ├── comparison/                # side-by-side comparison PNGs
│   ├── comparison_results.csv
│   ├── srcnn_training_log.csv
│   └── edsr_training_log.csv
├── app.py                        # Streamlit demo
└── requirements.txt
```

## 2. Setup

```bash
pip install -r requirements.txt
```

## 3. Get real chromosome images

Put your high-resolution chromosome images into:

```
dataset/train/HR/
dataset/val/HR/
dataset/test/HR/
```

**Check the dataset's license and permitted research use before using it.**
Filenames should be consistent — the same filename in `HR/` and `LR/`
represents the same underlying chromosome.

(If you don't have a dataset yet and just want to confirm the pipeline
works, run `preprocessing/generate_synthetic_dataset.py` — see step 4.)

## 4. Generate LR / medium-resolution training pairs

```bash
# Optional: only if you want to smoke-test without real data first
python preprocessing/generate_synthetic_dataset.py --out_dir dataset \
    --n_train 40 --n_val 8 --n_test 8 --hr_size 256

# Real pipeline step: generate LR images from your HR images
python preprocessing/create_lr_images.py \
    --hr_dir dataset/train/HR --lr_dir dataset/train/LR \
    --scale 4 --blur --noise --jpeg
python preprocessing/create_lr_images.py \
    --hr_dir dataset/val/HR --lr_dir dataset/val/LR --scale 4 --blur --noise --jpeg
python preprocessing/create_lr_images.py \
    --hr_dir dataset/test/HR --lr_dir dataset/test/LR --scale 4 --blur --noise --jpeg
```

`--blur --noise --jpeg` are optional degradation flags (Improvement A in the
project plan — start without them for a simpler first baseline, add them
later to make the model more robust to realistic degradations).

## 5. Establish the classical baseline

Do this **before** touching any neural network — it's the number every
deep model must beat:

```bash
python evaluation/bicubic_baseline.py --test_hr dataset/test/HR --test_lr dataset/test/LR
```

## 6. Train SRCNN (baseline deep model)

```bash
python training/train.py --model srcnn --epochs 50 \
    --train_hr dataset/train/HR --train_lr dataset/train/LR \
    --val_hr dataset/val/HR --val_lr dataset/val/LR
```

## 7. Train EDSR (main model)

```bash
python training/train.py --model edsr --epochs 50 --scale 4 \
    --num_features 64 --num_res_blocks 16 \
    --train_hr dataset/train/HR --train_lr dataset/train/LR \
    --val_hr dataset/val/HR --val_lr dataset/val/LR
```

Useful flags for both: `--patch_size`, `--batch_size`, `--lr`, `--loss
{l1,mse}`. Start with `l1` and no GANs/perceptual losses — get a reliable
baseline first (Step 7 in the project plan).

Training logs (`outputs/{model}_training_log.csv`) record epoch, train
loss, val loss, val PSNR, val SSIM, and time per epoch — plot these with
Matplotlib for your report.

## 8. Evaluate: Bicubic vs SRCNN vs EDSR

```bash
python evaluation/evaluate.py --scale 4 \
    --test_hr dataset/test/HR --test_lr dataset/test/LR \
    --srcnn_ckpt checkpoints/srcnn_best.pth \
    --edsr_ckpt checkpoints/edsr_best.pth
```

This produces:
- `outputs/comparison_results.csv` — per-image and averaged PSNR/SSIM for
  all three methods (use the real numbers here — never fabricate results).
- `outputs/comparison/<name>_compare.png` — a Medium-Res | Bicubic | SRCNN
  | EDSR | Ground-Truth strip per test image, for visual inspection of
  banding patterns, boundaries, and artifacts.

## 9. Run the demo

```bash
streamlit run app.py
```

Upload a medium-resolution chromosome image, optionally upload the matching
ground-truth HR image to see PSNR/SSIM, and view the super-resolved output
side-by-side with processing time.

## 10. A note on scientific honesty

Super-resolution models can enhance apparent detail without that detail
being biologically real (hallucination). The demo explicitly displays this
caveat. When you write up results, frame the contribution as:

> "We investigate whether deep learning can reconstruct high-resolution
> chromosome imagery from medium-resolution microscopy data while
> maintaining structural fidelity" — not "we increase image size."

Always report measured PSNR/SSIM from your own experiments, and visually
inspect (not just numerically score) whether fine structures — banding
patterns, chromatid boundaries — are faithfully reconstructed or invented.

## 11. Suggested extensions (add only after the core pipeline works)

- **Realistic degradation**: combine blur + noise + JPEG compression
  (`create_lr_images.py` already supports this via flags).
- **Architecture comparison**: extend `evaluate.py` with additional models.
- **Downstream task**: feed SR output into a chromosome segmentation model
  and measure whether SR improves segmentation accuracy — a strong
  experimental extension for the viva.

## 12. Suggested 3-month plan

See the project write-up for the week-by-week breakdown (Month 1:
foundation + SRCNN, Month 2: EDSR + tuning, Month 3: evaluation + demo +
documentation).

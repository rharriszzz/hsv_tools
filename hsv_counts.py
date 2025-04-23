```# filepath: c:\Users\rharr\git\hsv_tools\hsv_counts.py
"""
hsv_counts.py

Takes an image path as an argument and displays:
  - the original image (left)
  - a bar graph (right) whose X‐axis is the sequence of
    (H, S_bucket, V_bucket) tuples sorted by count
    and Y‐axis is the pixel count for each tuple.

Uses:
  - opencv-python for image I/O and HSV conversion
  - numpy for fast counting
  - matplotlib for plotting
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parameters
SV_BUCKET_SIZE = 16    # bucket size for S and V (so bins = 256//16 = 16)
MIN_HUE_COUNT = 0      # ignore hues with total count ≤ this
MIN_SV_COUNT = 0       # ignore (S_bucket, V_bucket) counts ≤ this

def main():
    p = argparse.ArgumentParser(
        description="Count pixels by H, S‐bucket, V‐bucket and plot results"
    )
    p.add_argument("image", help="Path to input image")
    args = p.parse_args()

    # load image and convert to HSV
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Error: could not load '{args.image}'")
        return
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # split channels and flatten
    h = img_hsv[:, :, 0].ravel()       # 0..179
    s = img_hsv[:, :, 1].ravel()       # 0..255
    v = img_hsv[:, :, 2].ravel()       # 0..255

    # total counts per hue
    max_hue = 180
    hue_counts = np.bincount(h, minlength=max_hue)

    # bucket S and V
    s_bins = 256 // SV_BUCKET_SIZE
    v_bins = 256 // SV_BUCKET_SIZE
    s_idx = np.minimum(s // SV_BUCKET_SIZE, s_bins - 1)
    v_idx = np.minimum(v // SV_BUCKET_SIZE, v_bins - 1)

    # 3D histogram: shape (H, S_bin, V_bin)
    counts3d = np.zeros((max_hue, s_bins, v_bins), dtype=int)
    np.add.at(counts3d, (h, s_idx, v_idx), 1)

    # build sequence of (H, S_bucket, V_bucket, count)
    seq = []
    # sort hues by descending count, apply MIN_HUE_COUNT cutoff
    valid_hues = np.where(hue_counts > MIN_HUE_COUNT)[0]
    sorted_hues = valid_hues[np.argsort(-hue_counts[valid_hues])]
    for hue in sorted_hues:
        # flatten this hue's S/V bins
        flat = counts3d[hue].ravel()
        valid_idxs = np.where(flat > MIN_SV_COUNT)[0]
        sorted_idxs = valid_idxs[np.argsort(-flat[valid_idxs])]
        for idx in sorted_idxs:
            sb = idx // v_bins
            vb = idx % v_bins
            cnt = counts3d[hue, sb, vb]
            seq.append((hue, sb, vb, cnt))

    # --- plotting ---
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 2]}
    )

    # show original image (convert BGR→RGB)
    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.axis("off")

    # bar chart of counts
    counts = [item[3] for item in seq]
    labels = [f"H{item[0]} S{item[1]} V{item[2]}" for item in seq]
    x = np.arange(len(seq))
    ax_bar.bar(x, counts, align="center")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=90, fontsize="small")
    ax_bar.set_xlabel("(Hue, S_bucket, V_bucket)")
    ax_bar.set_ylabel("Pixel count")
    ax_bar.set_title("HSV bucket counts")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
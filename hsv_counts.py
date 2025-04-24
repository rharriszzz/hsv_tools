# filepath: c:\Users\rharr\git\hsv_tools\hsv_counts.py
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
BAR_WIDTH = 3          # width in pixels per HSV bucket in plot
MIN_HUE_COUNT = 0      # ignore hues with total count ≤ this
MIN_SV_COUNT = 0       # ignore (S_bucket, V_bucket) counts ≤ this

def report_hue_buckets(seq, hue_counts, total_pixels, hue_bucket, hbs, max_print=10):
    """
    Prints the first max_print sub‐buckets for a given hue_bucket.
    """
    count = hue_counts[hue_bucket]
    center = hue_bucket * hbs + hbs/2
    print(f"Hue bucket={hue_bucket} (center≈{center:.1f}°) fraction: {count/total_pixels:.2e}")
    title = f"First {max_print} S/V buckets for Hue bucket={hue_bucket}"
    print(f"{title} (Hu_b, S_b, V_b, count, frac, hue_frac):")
    printed = 0
    for hb, sb, vb, cnt in seq:
        if hb != hue_bucket:
            continue
        frac = cnt / total_pixels
        hue_frac = cnt / hue_counts[hb]
        print(f"Hb={hb}, Sb={sb}, Vb={vb}, count={cnt}, frac={frac:.2e}, hue_frac={hue_frac:.2e}")
        printed += 1
        if printed >= max_print:
            break

def main():
    p = argparse.ArgumentParser(
        description="Count pixels by H, S‐bucket, V‐bucket and plot results"
    )
    p.add_argument("image", help="Path to input image")
    p.add_argument("--hue-bucket-size", type=float, default=4.0,
                   help="Hue bucket width (≥1.0), default 3")
    p.add_argument("--sat-bucket-size", type=float, default=8.0,
                   help="Saturation bucket width (≥1.0), default 8")
    p.add_argument("--val-bucket-size", type=float, default=64.0,
                   help="Value bucket width (≥1.0), default 64")
    p.add_argument("--by_hue", action="store_true",
                   help="Group and sort by hue buckets (default)")
    p.add_argument("--by_bucket", action="store_true",
                   help="Sort all buckets by count (no hue grouping)")
    args = p.parse_args()
    # choose mode (default to by_hue)
    default_by_hue = True
    use_by_hue = args.by_hue
    use_by_bucket = args.by_bucket
    if use_by_hue and use_by_bucket:
        print("Error: cannot specify both --by_hue and --by_bucket")
        return
    if not use_by_hue and not use_by_bucket:
        use_by_hue = default_by_hue
    # bar width override for bucket mode
    bar_width = BAR_WIDTH if use_by_hue else 6

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

    # bucket sizes from args (floats ≥1.0)
    hbs = max(1.0, args.hue_bucket_size)
    sbs = max(1.0, args.sat_bucket_size)
    vbs = max(1.0, args.val_bucket_size)

    # number of bins in each dimension
    max_hue = 180
    h_bins = int(np.ceil(max_hue / hbs))
    s_bins = int(np.ceil(256  / sbs))
    v_bins = int(np.ceil(256  / vbs))

    # bucket indices
    h_idx = np.minimum((h / hbs).astype(int), h_bins - 1)
    s_idx = np.minimum((s / sbs).astype(int), s_bins - 1)
    v_idx = np.minimum((v / vbs).astype(int), v_bins - 1)

    # total counts per hue‐bucket
    hue_counts = np.bincount(h_idx, minlength=h_bins)

    # 3D histogram over bucketed H, S, V
    counts3d = np.zeros((h_bins, s_bins, v_bins), dtype=int)
    np.add.at(counts3d, (h_idx, s_idx, v_idx), 1)

    # build sequence of (H, S_bucket, V_bucket, count)
    seq = []
    if use_by_hue:
        # group by hue then S/V
        valid_hues = np.where(hue_counts > MIN_HUE_COUNT)[0]
        sorted_hues = valid_hues[np.argsort(-hue_counts[valid_hues])]
        for hue in sorted_hues:
            flat = counts3d[hue].ravel()
            valid_idxs = np.where(flat > MIN_SV_COUNT)[0]
            sorted_idxs = valid_idxs[np.argsort(-flat[valid_idxs])]
            for idx in sorted_idxs:
                sb = idx // v_bins
                vb = idx % v_bins
                cnt = counts3d[hue, sb, vb]
                seq.append((hue, sb, vb, cnt))
    else:
        # flatten all buckets and sort purely by count
        for hue in range(h_bins):
            for sb in range(s_bins):
                for vb in range(v_bins):
                    cnt = counts3d[hue, sb, vb]
                    if cnt > MIN_SV_COUNT:
                        seq.append((hue, sb, vb, cnt))
        seq.sort(key=lambda x: x[3], reverse=True)

    # compute total pixels once
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]

    if use_by_hue:
        # report for top 6 hues (5 sub‑buckets each)
        valid_hues = np.where(hue_counts > MIN_HUE_COUNT)[0]
        sorted_hues = valid_hues[np.argsort(-hue_counts[valid_hues])]
        for hue in sorted_hues[:6]:
            report_hue_buckets(seq, hue_counts, total_pixels,
                               hue_bucket=hue, hbs=hbs, max_print=5)

    # drop buckets with hue_fraction < 1e-3
    seq = [item for item in seq
           if (item[3] / hue_counts[item[0]]) >= 1e-3]
    # cap total buckets so plot width ≃ image width
    max_bars = img_bgr.shape[1] // bar_width
    if len(seq) > max_bars:
        seq = seq[:max_bars]

    # --- plotting ---
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 2]}
    )

    # show original image (convert BGR→RGB)
    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.axis("off")

    # build RGB lookup for each bucket before plotting
    from matplotlib import colors
    hsv_mid = []
    # each h,sb,vb is the integer bucket index; use hbs,sbs,vbs to find the bucket center
    for h, sb, vb, _ in seq:
        h_mid = h * hbs + hbs / 2
        s_mid = sb * sbs + sbs / 2
        v_mid = vb * vbs + vbs / 2
        # normalize: H/180, S/255, V/255 → all in [0,1]
        hsv_mid.append([h_mid / 180.0, s_mid / 255.0, v_mid / 255.0])
    hsv_arr = np.array(hsv_mid, dtype=float).reshape(1, -1, 3)
    # rgb_arr2d shape = (N,3), values now guaranteed in 0–1
    rgb_arr2d = colors.hsv_to_rgb(hsv_arr)[0]

    # prepare pixel‐fraction data and horizontal positions
    fractions = [item[3] / total_pixels for item in seq]
    x = np.arange(len(fractions)) * bar_width

    # color bars by HSV→RGB but force very bright (near‐white) to black
    brightness = rgb_arr2d.sum(axis=1)
    # any bar whose RGB sum > 2.5 → nearly white → draw as black
    bar_colors = np.where(brightness[:, None] > 2.5, [0, 0, 0], rgb_arr2d)
    ax_bar.bar(x, fractions, width=bar_width, color=bar_colors, align='edge')
    ax_bar.set_yscale('log')

    ax_bar.margins(x=0)
    ax_bar.set_xlim(0, len(fractions) * bar_width)

    # remove text labels—use a colored strip instead
    ax_bar.set_xticks([])
    ax_bar.set_xticklabels([])

    # build a colored strip legend, each bucket repeated BAR_WIDTH cols
    strip_height = 20   # pixels tall for the HSV strip
    strip = np.repeat(rgb_arr2d[None, :, :], bar_width, axis=1)
    strip = np.tile(strip, (strip_height, 1, 1))

    # inset a new axis below ax_bar for the color strip
    ax_strip = ax_bar.inset_axes([0, -0.18, 1, 0.15], transform=ax_bar.transAxes)
    ax_strip.imshow(strip, aspect='auto')
    ax_strip.axis("off")

    # remove horizontal axis text
    # ax_bar.set_xlabel("(Hue, S_bucket, V_bucket)")
    ax_bar.set_ylabel("Pixel fraction")
    ax_bar.set_title("HSV bucket counts")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
# Figure provenance and reproduction

Run from the repository root after installing `requirements.txt`:

```bash
python scripts/generate_hsv_figures.py
python -m unittest discover -s tests -v
```

The generator also works when invoked by absolute path from another directory. It checks the source SHA-256 before calculating anything. No network or GUI is required for generation. Package versions, mask hashes, union counts, and the actual histogram bar fractions are recorded in [measurements.json](figures/measurements.json). For the reviewed rendering, install `requirements-figures.txt` in a separate virtual environment (Python 3.14 was used).

## Original photograph

- Source: [`rharriszzz/beads`, `beads-photo-2.jpg`, commit `020303ec16c81cb62802b6ae718adda9bbc2fdfa`](https://github.com/rharriszzz/beads/blob/020303ec16c81cb62802b6ae718adda9bbc2fdfa/beads-photo-2.jpg).
- Committed copy: [figures/source/beads-photo-2.jpg](figures/source/beads-photo-2.jpg), 1,291,036 bytes, 2540 × 3182 pixels.
- SHA-256: `eb7c9edb62f5580ef56632872da48da92556d62b758295137068cc2404dc8fbb`.
- Git blob SHA-1: `492a1cbb184eb31f1d54fba063e680b23d8510c9` (also checked against the GitHub source tree).

The copy comes directly from the committed Git blob, without re-encoding. It matches the original photograph in `fft-image-explorer` as well. No white balance, contrast adjustment, rotation, reconstruction, or generated imagery is used.

## Picker calculation and candidates

Coordinates below are integer pixels in the full photograph, origin at the upper left, x rightward and y downward. All three candidates display the same half-open crop: **x = [1640, 2020), y = [250, 690)**, or 380 × 440 pixels. Masks are computed on the full source image before cropping.

| Candidate | Line endpoints (x, y) | ΔH | ΔS | ΔV | Selected pixels, full image |
| --- | --- | ---: | ---: | ---: | ---: |
| A · Red, preferred | (1734, 325) → (1737, 342) | 5 | 45 | 65 | 110,151 |
| B · Yellow | (1888, 480) → (1896, 468) | 5 | 40 | 65 | 100,974 |
| C · Dark bead | (1798, 400) → (1804, 402) | 12 | 70 | 25 | 261,740 |

OpenCV converts the decoded BGR photograph with `COLOR_BGR2HSV`: H is 0–179 (one unit is approximately 2 degrees), S and V are 0–255. Hue distance is `min(abs(H1-H2), 180-abs(H1-H2))`; S and V use ordinary absolute differences. All comparisons include their tolerance boundary, with integer conversion before subtraction.

`hsv_picker.mask_from_line` is extracted from the application's original callback. The interactive application and generator both call it. It rasterizes the one-pixel sampling line with `cv2.line`, visits its pixels in array/raster order, and places each HSV sample into the first existing representative bucket within all three tolerances. Otherwise it starts a new bucket. The representatives remain unchanged; they are not averages. The final selection is the union of full-image matches to those representatives, using the same tolerances. This is more specific than matching every sampled color independently.

Selected pixels are **0 (black)** and unselected pixels **255 (white)**. There is no cleanup, morphology, smoothing, bead detection, or added selection. The three committed `mask-*.png` files are full-resolution lossless outputs. Union counts above count each pixel once; the original console's sum of per-bucket counts can count overlaps more than once.

For display, the photograph crop is resized to 760 × 880 with Pillow Lanczos resampling. The mask uses nearest-neighbor resampling to retain exact black/white values. Both panels have identical scale, crop, and orientation. Cyan endpoints and a line annotate the actual sampled coordinates on the photo; this annotation is added by the figure script, not drawn by the interactive application's window. The labels, border, and annotation are presentation elements, not a fabricated screenshot. Previews resize the completed 1600 × 1000 figures to 400 × 250 with Lanczos. Text uses Matplotlib's bundled DejaVu Sans.

## Histogram

The script executes `hsv_counts.py` with its real CLI parser, intercepting `plt.show()` to save the resulting plot:

```bash
python hsv_counts.py docs/figures/source/beads-photo-2.jpg \
  --hue-bucket-size 12 --sat-bucket-size 128 --val-bucket-size 128 --by_bucket
```

The input is the **entire, unresized photograph**: 8,082,280 pixels. H/S/V bucket indices use floor division by the supplied widths. These widths give 15 × 2 × 2 buckets. Nonempty buckets are sorted by decreasing population, then the application's original filter retains a bucket only if it contains at least 0.001 of its hue group's pixels. The width cap is 2540 // 6 = 423 bars and does not apply here. The result has **27 bars**, covering 8,082,227 pixels; 53 pixels are omitted by that filter. The plotted fractions retain the full-image denominator.

The generator preserves the application's bar heights, order, bucket-center colors, near-white-to-black rule (RGB sum > 2.5), color strip, and logarithmic scale. It adjusts figure dimensions, labels, panel positions, and grid, and crops only the displayed photograph to the picker viewport. This does not change the histogram's input. [hsv-counts-raw.png](figures/hsv-counts-raw.png) is the unstyled capture of the original plot; [hsv-counts-example.png](figures/hsv-counts-example.png) is the README composition.

## Review and validation

[Candidate comparison](figures/hsv-tools-candidates.png) includes all three picker figures and the histogram. Individual full-size candidates and 400 × 250 previews are committed alongside it.

The red candidate gives the clearest correspondence: several separated red bead bodies become visible black shapes, while yellow beads and most of the background stay unselected. The yellow candidate has a smaller, localized selection and excessive empty mask area. The dark sample selects more pixels but produces less distinct, sometimes connected shapes. The histogram explains color populations well at README size, but its small labels and logarithmic scale are less immediate at project-card size. The tight crop, visible sample, and single binary selection also distinguish this card from Bead Map's wider multicolor mask figure. The preferred card is a proposal for review, not a final selection for the central index.

All candidates were opened at full resolution and at 400 pixels wide. The actual OpenCV application was run on the original photograph for each sample, with real trackbars and callback selections automated through its display loop. Every resulting full-resolution mask matched the generated mask byte-for-byte. The same three masks were separately checked against the pre-refactor callback from commit `3e70fe3`. Tests cover hue wrapping, inclusive boundaries, signed subtraction, rasterized line interiors, greedy grouping, zoom/merge behavior, full-image regression masks, and an independent check of the actual histogram populations.

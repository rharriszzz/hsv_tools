#!/usr/bin/env python3
"""Regenerate authentic picker composites and the actual hsv_counts plot.

Run from any working directory. See docs/figures.md for provenance and settings.
No GUI or network access is needed.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsv_picker import mask_from_line

OUT = ROOT / 'docs/figures'
SOURCE = OUT / 'source/beads-photo-2.jpg'
SOURCE_SHA256 = 'eb7c9edb62f5580ef56632872da48da92556d62b758295137068cc2404dc8fbb'
CROP = (1640, 250, 2020, 690)  # x0, y0, x1 exclusive, y1 exclusive
SAMPLES = {
    'red': dict(line=((1734, 325), (1737, 342)), tolerances=(5, 45, 65)),
    'yellow': dict(line=((1888, 480), (1896, 468)), tolerances=(5, 40, 65)),
    'black': dict(line=((1798, 400), (1804, 402)), tolerances=(12, 70, 25)),
}
PREFERRED = 'red'
HISTOGRAM_ARGS = ['--hue-bucket-size', '12', '--sat-bucket-size', '128',
                  '--val-bucket-size', '128', '--by_bucket']
INK = '#172c40'
CYAN = '#65efff'
FONT = str(Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf')


def font(size):
    return ImageFont.truetype(FONT, size)


def picker_figure(rgb, mask, sample):
    """Equal, aligned crops; exact binary mask; line overlay only on the photo."""
    figure = Image.new('RGB', (1600, 1000), 'white')
    photo = Image.fromarray(rgb).crop(CROP).resize((760, 880), Image.Resampling.LANCZOS)
    binary = Image.fromarray(mask).crop(CROP).resize((760, 880), Image.Resampling.NEAREST)
    figure.paste(photo, (28, 70))
    figure.paste(binary, (812, 70))
    draw = ImageDraw.Draw(figure)
    draw.rectangle((811, 69, 1572, 950), outline='#dce1e5', width=1)
    points = [(28 + (x-CROP[0])*2 + .5, 70 + (y-CROP[1])*2 + .5)
              for x, y in sample['line']]
    draw.line(points, fill=INK, width=12)
    draw.line(points, fill=CYAN, width=7)
    for x, y in points:
        draw.ellipse((x-7, y-7, x+7, y+7), fill=CYAN, outline=INK, width=2)
    draw.text((28, 15), 'Sample a line', fill=INK, font=font(34))
    draw.text((812, 15), 'Matching HSV pixels', fill=INK, font=font(34))
    h, s, v = sample['tolerances']
    draw.text((28, 958), f'ΔH {h}  ·  ΔS {s}  ·  ΔV {v}', fill=INK, font=font(27))
    draw.text((812, 958), 'Black = selected', fill=INK, font=font(27))
    return figure


def counts_figure():
    """Run the actual CLI; retain its bars, filtering, colors and log scale."""
    captured = {}

    def save_actual_plot():
        fig = plt.gcf()
        captured['figure'] = fig
        # Save an unstyled capture as well, so the presentation is auditable.
        fig.savefig(OUT / 'hsv-counts-raw.png', dpi=150)

    argv = [str(ROOT/'hsv_counts.py'), str(SOURCE), *HISTOGRAM_ARGS]
    with patch.object(sys, 'argv', argv), patch.object(plt, 'show', save_actual_plot):
        with redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT/'hsv_counts.py'), run_name='__main__')
    fig = captured['figure']
    image_ax, bars_ax = fig.axes
    # main() locals are not returned by runpy. Capture numerical data directly
    # from the actual bar artists, rather than calculating another histogram.
    bars = bars_ax.patches
    values = [float(b.get_height()) for b in bars]
    fig.set_size_inches(16, 10)
    fig.set_facecolor('white')
    image_ax.set_position([.025, .25, .34, .63])
    image_ax.set_xlim(CROP[0]-.5, CROP[2]-.5)
    image_ax.set_ylim(CROP[3]-.5, CROP[1]-.5)
    image_ax.set_title('Photograph · detail', loc='left', fontsize=20, color=INK, pad=18)
    bars_ax.set_position([.43, .25, .545, .63])
    bars_ax.set_title('', loc='center')
    bars_ax.set_title('HSV populations · full photograph', loc='left', fontsize=21, color=INK, pad=20)
    bars_ax.set_ylabel('Pixel fraction (log scale)', fontsize=17, color=INK)
    bars_ax.tick_params(axis='y', labelsize=15, colors=INK)
    for side in ('top', 'right'):
        bars_ax.spines[side].set_visible(False)
    bars_ax.grid(axis='y', color='#dce1e5', linewidth=.7)
    bars_ax.set_axisbelow(True)
    fig.text(.43, .065, 'Buckets: H 12  ·  S 128  ·  V 128   |   Largest populations first',
             fontsize=17, color=INK)
    fig.text(.43, .025, 'Color strip = bucket centers; near-white bars are drawn black.',
             fontsize=15, color=INK)
    fig.savefig(OUT/'hsv-counts-example.png', dpi=100)
    plt.close(fig)
    return dict(arguments=HISTOGRAM_ARGS, input='full source photograph',
                bars=len(bars), displayed_pixel_fraction=sum(values),
                bar_pixel_fractions=values)


def comparison():
    sheet = Image.new('RGB', (1600, 1160), '#f2f4f5')
    draw = ImageDraw.Draw(sheet)
    entries = [('A · Red sample — preferred', 'candidate-red.png'),
               ('B · Yellow sample', 'candidate-yellow.png'),
               ('C · Dark-bead sample', 'candidate-black.png'),
               ('D · HSV Counts', 'hsv-counts-example.png')]
    for i, (label, filename) in enumerate(entries):
        x, y = 20+(i % 2)*800, 16+(i // 2)*580
        draw.text((x, y), label, fill=INK, font=font(29))
        with Image.open(OUT/filename) as im:
            sheet.paste(im.resize((760, 475), Image.Resampling.LANCZOS), (x, y+48))
    sheet.save(OUT/'hsv-tools-candidates.png')


def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'Source photograph hash mismatch: {digest}')
    bgr = cv2.imread(str(SOURCE))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    record = dict(source_sha256=digest, source_size=[rgb.shape[1], rgb.shape[0]],
                  crop_xyxy=CROP, preferred=PREFERRED, candidates={})
    for name, sample in SAMPLES.items():
        mask = mask_from_line(hsv, *sample['line'], sample['tolerances'])
        # Committed lossless masks make the calculation independently inspectable.
        cv2.imwrite(str(OUT/f'mask-{name}.png'), mask)
        figure = picker_figure(rgb, mask, sample)
        figure.save(OUT/f'candidate-{name}.png')
        preview = figure.resize((400, 250), Image.Resampling.LANCZOS)
        preview.save(OUT/f'candidate-{name}-preview.png')
        if name == PREFERRED:
            figure.save(OUT/'hsv-tools-card.png')
            preview.save(OUT/'hsv-tools-card-preview.png')
        record['candidates'][name] = dict(**sample, selected_pixels=int((mask == 0).sum()),
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    record['histogram'] = counts_figure()
    with Image.open(OUT/'hsv-counts-example.png') as im:
        im.resize((400, 250), Image.Resampling.LANCZOS).save(OUT/'hsv-counts-preview.png')
    comparison()
    record['versions'] = dict(python=sys.version.split()[0], opencv=cv2.__version__,
                              numpy=np.__version__, matplotlib=matplotlib.__version__,
                              pillow=Image.__version__)
    (OUT/'measurements.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Regenerated figures in {OUT}')


if __name__ == '__main__':
    main()

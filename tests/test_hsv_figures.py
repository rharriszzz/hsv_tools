"""Numerical checks for the shared picker and figures (no display required)."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import hsv_picker as picker
from scripts import generate_hsv_figures as figures


class PickerTests(unittest.TestCase):
    def test_circular_hue_and_inclusive_tolerance(self):
        hsv = np.array([[[h, 200, 200] for h in (0, 179, 1, 178, 2, 90)]], np.uint8)
        mask = picker.mask_from_line(hsv, (0, 0), (0, 0), (1, 0, 0))
        np.testing.assert_array_equal(mask, [[0, 0, 0, 255, 255, 255]])
        self.assertEqual(mask.dtype, np.uint8)

    def test_sv_boundaries_and_unsigned_subtraction(self):
        hsv = np.array([[[0, 5, 5], [179, 0, 0], [1, 10, 10],
                         [0, 11, 5], [0, 5, 11], [0, 255, 255]]], np.uint8)
        np.testing.assert_array_equal(
            picker.mask_from_line(hsv, (0, 0), (0, 0), (1, 5, 5)),
            [[0, 0, 0, 255, 255, 255]])

    def test_greedy_representatives_are_not_all_line_pixels(self):
        hsv = np.array([[[h, 200, 200] for h in (10, 12, 14)]], np.uint8)
        # H=12 joins H=10's bucket; it must not become a representative
        # that would independently select H=14. Endpoint order is raster order.
        for p1, p2 in [((0, 0), (1, 0)), ((1, 0), (0, 0))]:
            np.testing.assert_array_equal(
                picker.mask_from_line(hsv, p1, p2, (2, 0, 0)), [[0, 0, 255]])

    def test_diagonal_line_includes_interior_and_both_endpoints(self):
        hsv = np.full((3, 3, 3), [90, 200, 200], np.uint8)
        for i, h in enumerate((10, 20, 30)):
            hsv[i, i, 0] = h
        expected = np.where(np.eye(3, dtype=bool), 0, 255)
        np.testing.assert_array_equal(
            picker.mask_from_line(hsv, (0, 0), (2, 2), (0, 0, 0)), expected)

    def test_callback_zoom_merge_and_replace(self):
        hsv = np.array([[[10, 200, 200], [20, 200, 200], [30, 200, 200]]], np.uint8)
        with patch.multiple(picker, clicks=[], custom_mask=np.array([[0, 255, 255]], np.uint8),
                            merge_mode=True, zoom=2., offset_x=1, offset_y=0), \
             patch.object(cv2, 'getTrackbarPos', return_value=0), \
             redirect_stdout(io.StringIO()):
            for _ in range(2):
                picker.on_mouse(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, (hsv, 3, 1))
            np.testing.assert_array_equal(picker.custom_mask, [[0, 0, 255]])
            self.assertEqual(picker.clicks, [])
            picker.merge_mode = False
            for _ in range(2):
                picker.on_mouse(cv2.EVENT_LBUTTONDOWN, 2, 0, 0, (hsv, 3, 1))
            np.testing.assert_array_equal(picker.custom_mask, [[255, 255, 0]])


class FigureDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hsv = cv2.cvtColor(cv2.imread(str(figures.SOURCE)), cv2.COLOR_BGR2HSV)

    def test_original_photo_and_full_image_mask_regressions(self):
        self.assertEqual(hashlib.sha256(figures.SOURCE.read_bytes()).hexdigest(),
                         figures.SOURCE_SHA256)
        # Verified byte-for-byte against the original, pre-refactor callback.
        hashes = {
            'red': 'a545b44eed7ed52b412d466907c762dfb71ee095e491a8098bec61f2283409e5',
            'yellow': 'b072b8019b8f8a38e71920028f649e6d08464ea63b24a9722cfb6158d10a65d5',
            'black': '378ba1fefcb7b5d3c43dc4d3dd037dc4669a3466d348c83d490404518121350a',
        }
        for name, sample in figures.SAMPLES.items():
            with self.subTest(name=name):
                mask = picker.mask_from_line(self.hsv, *sample['line'], sample['tolerances'])
                self.assertEqual(hashlib.sha256(mask.tobytes()).hexdigest(), hashes[name])
                saved = cv2.imread(str(figures.OUT/f'mask-{name}.png'), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(mask, saved)

    def test_actual_histogram_fractions_against_independent_counts(self):
        h, s, v = [self.hsv[:, :, i].astype(np.uint16) for i in range(3)]
        packed = (h//12)*4 + (s//128)*2 + v//128
        counts = np.bincount(packed.ravel(), minlength=60)
        hues = counts.reshape(15, 4).sum(axis=1)
        keep = (counts > 0) & (counts >= np.repeat(hues, 4)*1e-3)
        expected = np.sort(counts[keep])[::-1] / h.size
        with tempfile.TemporaryDirectory() as directory, patch.object(figures, 'OUT', Path(directory)):
            actual = figures.counts_figure()
        np.testing.assert_allclose(actual['bar_pixel_fractions'], expected, rtol=0, atol=0)
        self.assertEqual(actual['bars'], 27)


if __name__ == '__main__':
    unittest.main()

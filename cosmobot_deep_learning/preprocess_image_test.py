import numpy as np
import pytest

from . import preprocess_image as module


# Use to quickly mock RGB pixels
def rgb_px(n):
    return [f'{px}{n}' for px in ['r', 'g', 'b']]


class TestCropSquare:
    # fmt: off
    @pytest.mark.parametrize(
        "name, image, expected",
        [
            (
                "no-op for square",
                np.array([
                    [rgb_px(1), rgb_px(2)],
                    [rgb_px(3), rgb_px(4)],
                ]),
                np.array([
                    [rgb_px(1), rgb_px(2)],
                    [rgb_px(3), rgb_px(4)],
                ]),
            ),
            (
                "crops vertical",
                np.array([
                    [rgb_px(1), rgb_px(3)],
                    [rgb_px(3), rgb_px(4)],
                    [rgb_px(5), rgb_px(6)],
                    [rgb_px(7), rgb_px(8)],
                ]),
                np.array([
                    [rgb_px(3), rgb_px(4)],
                    [rgb_px(5), rgb_px(6)],
                ]),
            ),
            (
                "crops horizontal",
                np.array([
                    [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
                    [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
                ]),
                np.array([
                    [rgb_px(2), rgb_px(3)],
                    [rgb_px(6), rgb_px(7)],
                ]),
            ),
            (
                "prefers top when offset",
                np.array([
                    [rgb_px(1), rgb_px(2)],
                    [rgb_px(3), rgb_px(4)],
                    [rgb_px(5), rgb_px(6)],
                ]),
                np.array([
                    [rgb_px(1), rgb_px(2)],
                    [rgb_px(3), rgb_px(4)],
                ]),
            ),
            (
                "prefers left when offset",
                np.array([
                    [rgb_px(1), rgb_px(2), rgb_px(3)],
                    [rgb_px(4), rgb_px(5), rgb_px(6)],
                ]),
                np.array([
                    [rgb_px(1), rgb_px(2)],
                    [rgb_px(4), rgb_px(5)],
                ]),
            ),
        ],
    )
    # fmt: on
    def test_crop_square(self, name, image, expected):
        actual = module.crop_square(image)
        np.testing.assert_array_equal(actual, expected)


class TestCropAndScaleImage:
    # TODO
    pass


class TestOpenCropAndScaleImage:
    # TODO
    pass


class TestSeriesOfImagesToNdarray:
    # TODO
    pass
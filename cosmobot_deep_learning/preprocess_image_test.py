import numpy as np
import pytest

from . import preprocess_image as module


# Use to quickly mock RGB pixels
def rgb_px(n):
    return [px * n for px in [1, 2, 3]]


class TestCropSquare:
    @pytest.mark.parametrize(
        "name, image, expected",
        [
            # fmt: off
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
            # fmt: on
        ],
    )
    def test_crop_square(self, name, image, expected):
        actual = module.crop_square(image)
        np.testing.assert_array_equal(actual, expected)


class TestCropAndScaleImage:
    rgb_image = np.array(
        [
            [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
            [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
            [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
            [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
            [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
            [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
        ],
        # cv2 requires dtype to be "uint8"
        dtype="uint8",
    )

    def test_doesnt_resize_if_crop_matches_output_size(self):
        expected = np.array(
            [
                [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
                [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
                [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
                [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
            ]
        )

        actual = module.crop_and_scale_image(self.rgb_image, output_size=4)

        np.testing.assert_array_equal(actual, expected)

    def test_resizes_to_output_size(self):
        # The specific numbers are from cv2's resize
        # fmt: off
        expected = np.array(
            [
                [[4, 7, 11], [6, 11, 17]],
                [[4, 7, 11], [6, 11, 17]]
            ]
        )
        # fmt: on

        actual = module.crop_and_scale_image(self.rgb_image, output_size=2)
        print(actual)

        np.testing.assert_array_equal(actual, expected)

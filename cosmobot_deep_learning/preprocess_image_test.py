from collections import OrderedDict

import numpy as np
import pytest

from . import preprocess_image as module


# Use to quickly mock RGB pixels
def rgb_px(n):
    return [px * n for px in [1, 2, 3]]


test_rgb_image = np.array(
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
    def test_doesnt_resize_if_crop_matches_output_size(self):
        expected = np.array(
            [
                [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
                [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
                [rgb_px(5), rgb_px(6), rgb_px(7), rgb_px(8)],
                [rgb_px(1), rgb_px(2), rgb_px(3), rgb_px(4)],
            ]
        )

        actual = module.crop_and_scale_image(test_rgb_image, output_size=4)

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

        actual = module.crop_and_scale_image(test_rgb_image, output_size=2)
        print(actual)

        np.testing.assert_array_equal(actual, expected)


class TestOpenAndPreprocessImages:
    def test_parallel_map_preserves_order(self, mocker):
        # Mostly a smoke test, but also a valuable regression test if
        # the map function is changed.

        images_to_open = OrderedDict(
            [
                ("image-0", np.array([[rgb_px(1), rgb_px(2)], [rgb_px(3), rgb_px(4)]])),
                ("image-1", np.array([[rgb_px(5), rgb_px(6)], [rgb_px(7), rgb_px(8)]])),
                ("image-2", np.array([[rgb_px(3), rgb_px(4)], [rgb_px(1), rgb_px(2)]])),
            ]
        )

        # Ensure the images are returned in whatever order they are requested
        mocker.patch.object(
            module, "open_as_rgb", side_effect=lambda x: images_to_open[x]
        )

        actual = module.open_and_preprocess_images(
            ["image-0", "image-1", "image-2"],
            image_size=2,  # Use settings to prevent any transformations from happening
            max_workers=2,
        )

        expected = np.array(
            [
                images_to_open["image-0"],
                images_to_open["image-1"],
                images_to_open["image-2"],
            ]
        )

        np.testing.assert_array_equal(actual, expected)

    def test_image_size_applied_correctly(self, mocker):
        mocker.patch.object(module, "open_as_rgb", return_value=test_rgb_image)

        # fmt: off
        expected = np.array(
            [
                [[4, 7, 11], [6, 11, 17]],
                [[4, 7, 11], [6, 11, 17]]
            ]
        )
        # fmt: on

        actual = module.open_and_preprocess_images(
            ["image-0"],
            image_size=2,  # Use settings to prevent any transformations from happening
            max_workers=1,
        )

        np.testing.assert_array_equal(actual[0], expected)

import cv2
import numpy as np


from picamraw import PiRawBayer, PiCameraVersion


RAW_BIT_DEPTH = 2 ** 10  # used for normalizing DN to DNR


# COPY-PASTA: This has been copied (and renamed) from cosmobot-process-experiment
def open_as_rgb(raw_image_path):
    """ Extracts the raw bayer data from a JPEG+RAW file and converts it to an
        `RGB Image` (see definition in README).

    Args:
        raw_image_path: The full path to the JPEG+RAW file

    Returns:
        An `RGB Image`
    """
    raw_bayer = PiRawBayer(
        filepath=raw_image_path, camera_version=PiCameraVersion.V2, sensor_mode=0
    )

    # Divide by the bit-depth of the raw data to normalize into the (0,1) range
    rgb_image = raw_bayer.to_rgb() / RAW_BIT_DEPTH

    return rgb_image


def crop_square(rgb_image):
    """ Crop an RGB image to a square, preserving as much of the center of the image as possible.

        Args:
            rgb_image: An `RGB Image`

        Returns:
            An `RGB Image`, cropped to a square around the center
    """
    height, width, depth = rgb_image.shape
    new_side_length = min(height, width)

    # The // operator performs integer division, rounding down
    start_col = (width - new_side_length) // 2
    start_row = (height - new_side_length) // 2

    return rgb_image[
        start_row : start_row + new_side_length, start_col : start_col + new_side_length
    ]


def crop_and_scale_image(rgb_image, output_size):
    """Call crop_square on an RGB image and resize it to the specified dimension. Returns a PIL image"""
    square_image = crop_square(rgb_image)
    return cv2.resize(square_image, (output_size, output_size))


def open_crop_and_scale_image(filepath, output_size):
    """Open an RGB image, crop and resize to the specified dimension. Returns a numpy array"""
    rgb_image = open_as_rgb(filepath)
    prepared_image = crop_and_scale_image(rgb_image, output_size)
    return np.array(prepared_image)


def series_of_images_to_ndarray(series):
    # This is silly, but I couldn't find a better way. When you get series.values(), you get an array of arrays, rather than one ndarray :\
    # There's probably a better way.
    return np.array([x for x in series])

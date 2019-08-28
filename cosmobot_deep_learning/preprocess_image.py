import sys
import functools
import concurrent.futures
import multiprocessing

import cv2
import numpy as np
from tqdm.auto import tqdm


from picamraw import PiRawBayer, PiCameraVersion


RAW_BIT_DEPTH = 2 ** 10  # used for normalizing DN to DNR


def fix_multiprocessing_with_keras_on_macos():
    """ The concurrent futures process pool used to process images doesn't like something about keras on MacOS
        Learn more about python fork/spawn trickiness here
        https://codewithoutrules.com/2018/09/04/python-multiprocessing/
        With python 3.7 this can be set per-process pool, rather than globally
    """
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn")


# COPY-PASTA: This has been copied (and renamed) from cosmobot-process-experiment
def open_as_rgb(raw_image_path: str):
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


def crop_square(rgb_image: np.ndarray):
    """ Crop an RGB image to a square, preserving as much of the center of the image as possible.

    Args:
        rgb_image: An `RGB Image`

    Returns:
        An `RGB Image`, cropped to a square around the center
    """
    height, width, depth = rgb_image.shape
    new_side_length = min(height, width)

    start_col = (width - new_side_length) // 2
    start_row = (height - new_side_length) // 2

    return rgb_image[
        start_row : start_row + new_side_length, start_col : start_col + new_side_length
    ]


def crop_and_scale_image(rgb_image: np.ndarray, output_size: int):
    """ Call crop_square on an RGB image and resize it to the specified dimension. Returns a PIL image

    Args:
        rgb_image: An `RGB Image`
        output_size: The number

    Returns:
        An `RGB Image`, cropped to a square around the center
    """
    square_image = crop_square(rgb_image)
    cv2_image = cv2.resize(square_image, (output_size, output_size))
    return np.array(cv2_image)


def open_crop_and_scale_image(raw_image_path: str, output_size: int):
    """ Opens a JPEG+RAW file as an `RGB Image`, then crops to a square and resizes
        to the desired ouput_size.

    Args:
        raw_image_path: The full path to a JPEG+RAW file
        output_size: The desired width and height (in pixels) of the square output image

    Returns:
        An `RGB Image`, cropped and scaled
    """
    rgb_image = open_as_rgb(raw_image_path)
    return crop_and_scale_image(rgb_image, output_size)


def open_and_preprocess_images(image_filepaths, image_size, max_workers=None):
    """ Preprocess the input images and prepare them for direct use in training a model.
        NOTE: The progress bar will only update sporadically.

        Args:
            image_filepaths: An iterable list of filepaths to images to prepare
            image_size: The desired side length of the output (square) image
            max_workers: Optional. Number of parallel processes to use to prepare images.
                Defaults to the number of CPU cores.
        Returns:
            A single numpy array of all images resized to the appropriate dimensions and concatenated
    """

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:

        # Use partial function to pass desired image_size through to new process
        open_crop_and_scale_image_with_size = functools.partial(
            open_crop_and_scale_image, output_size=image_size
        )

        return np.array(
            list(
                tqdm(
                    executor.map(
                        open_crop_and_scale_image_with_size,
                        image_filepaths,
                        chunksize=100,  # SWAG value, but much faster than the default
                    ),
                    total=len(image_filepaths),
                )
            )
        )

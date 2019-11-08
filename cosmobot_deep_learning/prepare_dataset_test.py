from unittest.mock import sentinel

import numpy as np
import pandas as pd
import pytest

from . import prepare_dataset as module


MOCK_DATASET = pd.DataFrame(
    {
        "numeric_input_column": [1, 2, 3],
        "other_column": [4, 5, 6],
        "another_column": [7, 8, 9],
        "DO_label_column": [10, 20, 30],
        "DO patch r_msorm": [4, 5, 6],
        "reference patch r_msorm": [7, 8, 9],
        "local_filepath": [sentinel.filepath] * 3,
        "training_resampled": [True, True, False],
        "dev": [False, False, True],
        "ROI definitions": ["{}", "{}", "{}"],
    }
)


class TestExtractInputs:
    def test_returns_dataset_slimmed_to_columns(self):
        expected = np.array([[1, 2, 3]]).T

        actual = module.extract_numeric_inputs(
            MOCK_DATASET, input_column_names=["numeric_input_column"]
        )

        np.testing.assert_array_equal(actual, expected)

    def test_returns_dataset_with_sr_column(self):
        expected = np.array(
            [
                # fmt: off
                [1, 2, 3],
                [4 / 7, 5 / 8, 6 / 9]
                # fmt: on
            ]
        ).T

        actual = module.extract_numeric_inputs(
            MOCK_DATASET, input_column_names=["numeric_input_column", "sr"]
        )

        np.testing.assert_array_equal(actual, expected)


class TestExtractLabels:
    def test_returns_single_column_as_2d_array(self):
        expected = np.array([[10, 20, 30]]).T

        actual = module.extract_labels(
            MOCK_DATASET, label_column="DO_label_column", label_scale_factor_mmhg=1
        )

        np.testing.assert_array_equal(actual, expected)

    def test_scales_by_scale_factor(self):
        scale_factor = 100
        expected = np.array(
            [[10 / scale_factor, 20 / scale_factor, 30 / scale_factor]]
        ).T

        actual = module.extract_labels(
            MOCK_DATASET,
            label_column="DO_label_column",
            label_scale_factor_mmhg=scale_factor,
        )

        np.testing.assert_array_equal(actual, expected)


class TestPrepareDatasetnumeric:
    def test_returns_expected_x_y(self):
        scale_factor = 100

        expected_x = np.array([[1, 2, 3], [4 / 7, 5 / 8, 6 / 9]]).T
        expected_y = np.array(
            [[10 / scale_factor, 20 / scale_factor, 30 / scale_factor]]
        ).T

        actual = module.prepare_dataset_numeric(
            MOCK_DATASET,
            {
                "numeric_input_columns": ["numeric_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
            },
        )

        expected = (expected_x, expected_y)

        for i, _ in enumerate(actual):
            np.testing.assert_array_equal(actual[i], expected[i])


@pytest.fixture
def mock_open_and_preprocess_images(mocker):
    def _mock_open_and_preprocess_images(filepaths, image_size):
        return np.array([sentinel.image] * len(filepaths))

    return mocker.patch.object(
        module,
        "open_and_preprocess_images",
        side_effect=_mock_open_and_preprocess_images,
    )


class TestPrepareDatasetImageAndnumeric:
    def test_returns_expected_x_y(self, mock_open_and_preprocess_images):
        scale_factor = 100

        expected_x_numeric = np.array([[1, 2, 3], [4 / 7, 5 / 8, 6 / 9]]).T
        expected_x_images = np.array([sentinel.image, sentinel.image, sentinel.image])
        expected_y = np.array(
            [[10 / scale_factor, 20 / scale_factor, 30 / scale_factor]]
        ).T

        actual = module.prepare_dataset_image_and_numeric(
            MOCK_DATASET,
            {
                "numeric_input_columns": ["numeric_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
                "image_size": sentinel.image_size,
            },
        )

        ((actual_x_numeric, actual_x_images), actual_y) = actual

        # No easy way to compare tuples of np arrays
        np.testing.assert_array_equal(actual_x_numeric, expected_x_numeric)
        np.testing.assert_array_equal(actual_x_images, expected_x_images)
        np.testing.assert_array_equal(actual_y, expected_y)


@pytest.fixture
def mock_open_and_preprocess_image_ROIs(mocker):
    def _mock_open_and_preprocess_image_ROIs(filepaths, ROI_names, image_size):
        # Return a list rather than an np.array for easier comparison
        return [[ROI_name] * len(filepaths) for ROI_name in ROI_names]

    return mocker.patch.object(
        module,
        "open_and_preprocess_image_ROIs",
        side_effect=_mock_open_and_preprocess_image_ROIs,
    )


class TestPrepareDatasetROIAndNumeric:
    def test_returns_expected_x_y(self, mock_open_and_preprocess_image_ROIs):
        scale_factor = 100
        mock_ROIs = [sentinel.ROI_0, sentinel.ROI_1]

        expected_x_numeric = np.array([[1, 2, 3], [4 / 7, 5 / 8, 6 / 9]]).T
        expected_x_ROIs = [
            [sentinel.ROI_0, sentinel.ROI_0, sentinel.ROI_0],
            [sentinel.ROI_1, sentinel.ROI_1, sentinel.ROI_1],
        ]
        expected_y = np.array(
            [[10 / scale_factor, 20 / scale_factor, 30 / scale_factor]]
        ).T

        actual = module.prepare_dataset_ROIs_and_numeric(
            MOCK_DATASET,
            {
                "numeric_input_columns": ["numeric_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
                "image_size": sentinel.image_size,
                "input_ROI_names": mock_ROIs,
            },
        )

        (actual_x, actual_y) = actual

        # Number of X values = 1 numeric + 1 for each ROI
        assert len(actual_x) == len(mock_ROIs) + 1
        assert actual_x[1:] == expected_x_ROIs

        # No easy way to compare tuples of np arrays
        np.testing.assert_array_equal(actual_x[0], expected_x_numeric)
        np.testing.assert_array_equal(actual_y, expected_y)

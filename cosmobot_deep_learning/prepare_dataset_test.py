from unittest.mock import sentinel

import numpy as np
import pandas as pd
import pytest

from . import prepare_dataset as module


MOCK_DATASET = pd.DataFrame(
    {
        "numerical_input_column": [1, 2, 3],
        "other_column": [4, 5, 6],
        "another_column": [7, 8, 9],
        "DO_label_column": [10, 20, 30],
        "DO patch r_msorm": [4, 5, 6],
        "reference patch r_msorm": [7, 8, 9],
        "local_filepath": [sentinel.filepath] * 3,
        "training_resampled": [True, True, False],
        "test": [False, False, True],
    }
)


class TestExtractInputs:
    def test_returns_dataset_slimmed_to_columns(self):
        expected = np.array([[1, 2, 3]]).T

        actual = module.extract_inputs(
            MOCK_DATASET, input_column_names=["numerical_input_column"]
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

        actual = module.extract_inputs(
            MOCK_DATASET, input_column_names=["numerical_input_column", "sr"]
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


class TestPrepareDatasetNumerical:
    def test_returns_expected_x_y_train_test(self):
        scale_factor = 100

        expected_x_train = np.array([[1, 2], [4 / 7, 5 / 8]]).T
        expected_y_train = np.array([[10 / scale_factor, 20 / scale_factor]]).T
        expected_x_test = np.array([[3], [6 / 9]]).T
        expected_y_test = np.array([[30 / scale_factor]]).T

        actual = module.prepare_dataset_numerical(
            MOCK_DATASET,
            {
                "numerical_input_columns": ["numerical_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
            },
        )

        expected = (
            expected_x_train,
            expected_y_train,
            expected_x_test,
            expected_y_test,
        )

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


class TestPrepareDatasetImageAndNumerical:
    def test_returns_expected_x_y_train_test(self, mock_open_and_preprocess_images):
        scale_factor = 100

        expected_x_train_numerical = np.array([[1, 2], [4 / 7, 5 / 8]]).T
        expected_x_train_images = np.array([sentinel.image, sentinel.image])
        expected_y_train = np.array([[10 / scale_factor, 20 / scale_factor]]).T
        expected_x_test_numerical = np.array([[3], [6 / 9]]).T
        expected_x_test_images = np.array([sentinel.image])
        expected_y_test = np.array([[30 / scale_factor]]).T

        actual = module.prepare_dataset_image_and_numerical(
            MOCK_DATASET,
            {
                "numerical_input_columns": ["numerical_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
                "image_size": sentinel.image_size,
            },
        )

        (
            (actual_x_train_numerical, actual_x_train_images),
            actual_y_train,
            (actual_x_test_numerical, actual_x_test_images),
            actual_y_test,
        ) = actual

        # No easy way to compare tuples of np arrays
        np.testing.assert_array_equal(
            actual_x_train_numerical, expected_x_train_numerical
        )
        np.testing.assert_array_equal(actual_x_train_images, expected_x_train_images)
        np.testing.assert_array_equal(actual_y_train, expected_y_train)
        np.testing.assert_array_equal(
            actual_x_test_numerical, expected_x_test_numerical
        )
        np.testing.assert_array_equal(actual_x_test_images, expected_x_test_images)
        np.testing.assert_array_equal(actual_y_test, expected_y_test)

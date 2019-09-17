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
        module.SETPOINT_O2_FRACTION_COLUMN_NAME: [1, 2, 3],
        module.SETPOINT_TEMPERATURE_COLUMN_NAME: [1, 2, 3],
    }
)


class TestExtractInputs:
    def test_returns_dataset_slimmed_to_columns(self):
        expected = np.array([[1, 2, 3]]).T

        actual = module.extract_inputs(
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

        actual = module.extract_inputs(
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
    def test_returns_expected_x_y_train_dev(self):
        scale_factor = 100

        expected_x_train = np.array([[1, 2], [4 / 7, 5 / 8]]).T
        expected_y_train = np.array([[10 / scale_factor, 20 / scale_factor]]).T
        expected_x_dev = np.array([[3], [6 / 9]]).T
        expected_y_dev = np.array([[30 / scale_factor]]).T

        actual = module.prepare_dataset_numeric(
            MOCK_DATASET,
            {
                "numeric_input_columns": ["numeric_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
                "training_set_column": "training_resampled",
                "dev_set_column": "dev",
            },
        )

        expected = (expected_x_train, expected_y_train, expected_x_dev, expected_y_dev)

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
    def test_returns_expected_x_y_train_dev(self, mock_open_and_preprocess_images):
        scale_factor = 100

        expected_x_train_numeric = np.array([[1, 2], [4 / 7, 5 / 8]]).T
        expected_x_train_images = np.array([sentinel.image, sentinel.image])
        expected_y_train = np.array([[10 / scale_factor, 20 / scale_factor]]).T
        expected_x_dev_numeric = np.array([[3], [6 / 9]]).T
        expected_x_dev_images = np.array([sentinel.image])
        expected_y_dev = np.array([[30 / scale_factor]]).T

        actual = module.prepare_dataset_image_and_numeric(
            MOCK_DATASET,
            {
                "numeric_input_columns": ["numeric_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
                "image_size": sentinel.image_size,
                "training_set_column": "training_resampled",
                "dev_set_column": "dev",
            },
        )

        (
            (actual_x_train_numeric, actual_x_train_images),
            actual_y_train,
            (actual_x_dev_numeric, actual_x_dev_images),
            actual_y_dev,
        ) = actual

        # No easy way to compare tuples of np arrays
        np.testing.assert_array_equal(actual_x_train_numeric, expected_x_train_numeric)
        np.testing.assert_array_equal(actual_x_train_images, expected_x_train_images)
        np.testing.assert_array_equal(actual_y_train, expected_y_train)
        np.testing.assert_array_equal(actual_x_dev_numeric, expected_x_dev_numeric)
        np.testing.assert_array_equal(actual_x_dev_images, expected_x_dev_images)
        np.testing.assert_array_equal(actual_y_dev, expected_y_dev)


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
    def test_returns_expected_x_y_train_dev(self, mock_open_and_preprocess_image_ROIs):
        scale_factor = 100
        mock_ROIs = [sentinel.ROI_0, sentinel.ROI_1]

        expected_x_train_numeric = np.array([[1, 2], [4 / 7, 5 / 8]]).T
        expected_x_train_ROIs = [
            [sentinel.ROI_0, sentinel.ROI_0],
            [sentinel.ROI_1, sentinel.ROI_1],
        ]
        expected_y_train = np.array([[10 / scale_factor, 20 / scale_factor]]).T
        expected_x_dev_numeric = np.array([[3], [6 / 9]]).T
        expected_x_dev_ROIs = [[sentinel.ROI_0], [sentinel.ROI_1]]
        expected_y_dev = np.array([[30 / scale_factor]]).T

        actual = module.prepare_dataset_ROIs_and_numeric(
            MOCK_DATASET,
            {
                "numeric_input_columns": ["numeric_input_column", "sr"],
                "label_column": "DO_label_column",
                "label_scale_factor_mmhg": scale_factor,
                "image_size": sentinel.image_size,
                "training_set_column": "training_resampled",
                "dev_set_column": "dev",
                "input_ROI_names": mock_ROIs,
            },
        )

        (actual_x_train, actual_y_train, actual_x_dev, actual_y_dev) = actual

        # Number of X values = 1 numeric + 1 for each ROI
        assert len(actual_x_train) == len(mock_ROIs) + 1
        assert actual_x_train[1:] == expected_x_train_ROIs
        assert actual_x_dev[1:] == expected_x_dev_ROIs

        # No easy way to compare tuples of np arrays
        np.testing.assert_array_equal(actual_x_train[0], expected_x_train_numeric)
        np.testing.assert_array_equal(actual_y_train, expected_y_train)
        np.testing.assert_array_equal(actual_x_dev[0], expected_x_dev_numeric)
        np.testing.assert_array_equal(actual_y_dev, expected_y_dev)


# fmt: off
@pytest.mark.parametrize(
    "input_arr, output_len, expected_output", (
        ([], 0, []),
        ([1], 1, [1]),
        ([1, 2], 1, [1]),
        ([1, 2, 3, 4, 5], 4, [1, 2, 3, 5]),
    )
)
# fmt: on
def test_uniformly_sample_array(input_arr, output_len, expected_output):
    result = module._uniformly_sample_array(np.array(input_arr), output_len)
    assert len(result) == output_len
    np.testing.assert_array_equal(result, expected_output)


def test_round_setpoint_columns():
    dataset = pd.DataFrame({
        module.SETPOINT_O2_FRACTION_COLUMN_NAME: [0.123456789, 0.12346],
        module.SETPOINT_TEMPERATURE_COLUMN_NAME: [0.123456789, 0.123],
    })
    expected_dataset = pd.DataFrame({
        module.SETPOINT_O2_FRACTION_COLUMN_NAME: [0.12346, 0.12346],
        module.SETPOINT_TEMPERATURE_COLUMN_NAME: [0.123, 0.123],
    })
    module._round_setpoint_columns(dataset)
    pd.testing.assert_frame_equal(dataset, expected_dataset)


class TestReduceImagesPerSetpoint():
    def test_one_image_per_setpoint(self):
        dataset = pd.DataFrame({
            module.SETPOINT_O2_FRACTION_COLUMN_NAME: [0.1, 0.1, 0.2, 0.2],
            module.SETPOINT_TEMPERATURE_COLUMN_NAME: [0.10, 0.10, 0.20, 0.30],
            'local_filepath': ['image1', 'image2', 'image3', 'image4'],
        })

        # image2 should be removed since it is at the same setpoint as image1
        expected_dataset = dataset[~(dataset['local_filepath'] == 'image2')]

        result = module._reduce_images_per_setpoint(dataset, max_images_per_setpoint=1)
        pd.testing.assert_frame_equal(result, expected_dataset)

    def test_max_two_images_per_setpoint(self):
        dataset = pd.DataFrame({
            module.SETPOINT_O2_FRACTION_COLUMN_NAME: [0.1, 0.1, 0.1, 0.2],
            module.SETPOINT_TEMPERATURE_COLUMN_NAME: [0.10, 0.10, 0.10, 0.30],
            'local_filepath': ['image1', 'image2', 'image3', 'image4'],
        })

        # image3 should be removed since it's the 3rd image for that setpoint
        expected_dataset = dataset[~(dataset['local_filepath'] == 'image3')]

        result = module._reduce_images_per_setpoint(dataset, max_images_per_setpoint=2)
        pd.testing.assert_frame_equal(result, expected_dataset)

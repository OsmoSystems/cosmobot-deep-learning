from unittest.mock import sentinel

import numpy as np
import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal

from . import run as module


@pytest.fixture
def mock_get_dataset_cache_filepath(mocker):
    return mocker.patch.object(module, "get_dataset_cache_filepath")


@pytest.fixture
def mock_dataset_cache_helpers(mocker):
    mock_get_prepared_dataset = mocker.patch.object(module, "_get_prepared_dataset")
    mock_load_dataset_cache = mocker.patch.object(module, "_load_dataset_cache")
    mock_save_dataset_cache = mocker.patch.object(module, "_save_dataset_cache")
    return {
        "_get_prepared_dataset": mock_get_prepared_dataset,
        "_load_dataset_cache": mock_load_dataset_cache,
        "_save_dataset_cache": mock_save_dataset_cache,
    }


class TestLoggableHyperparameters:
    def test_overrides_unloggable_metrics(self):
        def mock_custom_metric():
            return None

        unloggable_hyperparameters = {
            "some other attribute": sentinel.something,
            "metrics": ["a metric", "another metric", mock_custom_metric],
        }
        expected = {
            "some other attribute": sentinel.something,
            "metrics": ["a metric", "another metric", "mock_custom_metric"],
        }
        actual = module._loggable_hyperparameters(unloggable_hyperparameters)

        assert actual == expected


class TestDryRunFlag:
    def test_creates_tiny_dataset(self):
        mock_hyperparameters = {
            "training_set_column": "training_resampled",
            "dev_set_column": "test",
        }

        test_df = pd.DataFrame(
            {
                "training_resampled": [True, True, True, False, False, False],
                "test": [False, False, False, True, True, True],
            },
            index=[0, 1, 2, 3, 4, 5],
        )

        expected_df = pd.DataFrame(
            {
                "training_resampled": [True, True, False, False],
                "test": [False, False, True, True],
            },
            index=[0, 1, 3, 4],
        )

        actual_df = module._generate_tiny_dataset(test_df, mock_hyperparameters)

        pd.testing.assert_frame_equal(actual_df, expected_df)


class TestDatasetCache:
    def test_saves_new_dataset_cache_when_file_doesnt_exist(
        self, mocker, mock_get_dataset_cache_filepath, mock_dataset_cache_helpers
    ):
        mocker.patch("os.path.isfile", return_value=False)
        mock_dataset_cache_helpers[
            "_get_prepared_dataset"
        ].return_value = sentinel.test_dataset

        mock_get_dataset_cache_filepath.return_value = sentinel.cache_filepath

        actual_dataset = module._prepare_dataset_with_caching(
            prepare_dataset=sentinel.prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dryrun=sentinel.dryrun,
            dataset_cache_name="test",
        )

        mock_dataset_cache_helpers["_get_prepared_dataset"].assert_called_once_with(
            prepare_dataset=sentinel.prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dryrun=sentinel.dryrun,
        )
        assert actual_dataset == sentinel.test_dataset
        mock_dataset_cache_helpers["_load_dataset_cache"].assert_not_called()
        mock_dataset_cache_helpers["_save_dataset_cache"].assert_called_once_with(
            sentinel.cache_filepath, sentinel.test_dataset
        )

    def test_loads_existing_dataset_cache_when_file_exists(
        self, mocker, mock_get_dataset_cache_filepath, mock_dataset_cache_helpers
    ):
        mocker.patch("os.path.isfile", return_value=True)
        mock_get_dataset_cache_filepath.return_value = sentinel.cache_filepath
        mock_dataset_cache_helpers[
            "_get_prepared_dataset"
        ].return_value = sentinel.mock_dataset

        mock_dataset_cache_helpers[
            "_load_dataset_cache"
        ].return_value = sentinel.mock_dataset_cache

        actual_dataset = module._prepare_dataset_with_caching(
            prepare_dataset=sentinel.prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dryrun=sentinel.dryrun,
            dataset_cache_name="test",
        )

        mock_dataset_cache_helpers["_get_prepared_dataset"].assert_not_called()
        mock_dataset_cache_helpers["_load_dataset_cache"].assert_called_once_with(
            sentinel.cache_filepath
        )
        mock_dataset_cache_helpers["_save_dataset_cache"].assert_not_called()
        assert actual_dataset == sentinel.mock_dataset_cache

    def test_ignores_cache_when_no_name_specified(
        self, mocker, mock_get_dataset_cache_filepath, mock_dataset_cache_helpers
    ):
        mock_dataset_cache_helpers[
            "_get_prepared_dataset"
        ].return_value = sentinel.test_dataset

        actual_dataset = module._prepare_dataset_with_caching(
            prepare_dataset=sentinel.prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dryrun=sentinel.dryrun,
            dataset_cache_name=None,
        )

        mock_dataset_cache_helpers["_get_prepared_dataset"].assert_called_once_with(
            prepare_dataset=sentinel.prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dryrun=sentinel.dryrun,
        )
        mock_dataset_cache_helpers["_load_dataset_cache"].assert_not_called()
        mock_dataset_cache_helpers["_save_dataset_cache"].assert_not_called()
        assert actual_dataset == sentinel.test_dataset


class TestShuffleDataframe:
    @pytest.mark.parametrize(
        "dataframe,expected",
        (
            (pd.DataFrame(), pd.DataFrame()),
            (pd.DataFrame({"col_1": [0]}), pd.DataFrame({"col_1": [0]})),
            (
                pd.DataFrame({"col_1": [0, 1, 2, 3]}),
                pd.DataFrame({"col_1": [2, 3, 1, 0]}),
            ),
            (
                pd.DataFrame({"col_1": [0, 1, 2], "col_2": [10, 11, 12]}),
                pd.DataFrame({"col_1": [2, 1, 0], "col_2": [12, 11, 10]}),
            ),
        ),
    )
    def test_shuffle_dataframe(self, dataframe, expected):
        # reset the index so it has an equivalent index type to the result
        expected = expected.reset_index(drop=True)

        assert_frame_equal(module._shuffle_dataframe(dataframe), expected)


class TestSampleArrays:
    def test_samples_list_of_arrays(self):
        actual = module._sample_all_arrays_in_list(
            [np.zeros(5), np.ones(5)], sample_size=3
        )
        np.testing.assert_array_equal(actual[0], np.zeros(3))
        np.testing.assert_array_equal(actual[1], np.ones(3))

    def test_samples_single_array(self):
        actual = module._sample_all_arrays_in_list(np.zeros(5), sample_size=3)
        np.testing.assert_array_equal(actual, np.zeros(3))


class TestSampleArray:
    def test_samples_ndarray(self):
        # simulate an ndarray of 10 images
        ndarray = np.reshape(np.array(range(3000)), (10, 10, 10, 3))
        actual = module._sample_array(ndarray, sample_size=5)

        assert actual.shape == (5, 10, 10, 3)

    def test_samples_ndarray_at_max_len(self):
        # simulate an ndarray of 10 images
        ndarray = np.reshape(np.array(range(3000)), (10, 10, 10, 3))
        actual = module._sample_array(ndarray, sample_size=20)

        assert actual.shape == (10, 10, 10, 3)
        np.testing.assert_array_equal(np.sort(ndarray, axis=0), np.sort(actual, axis=0))

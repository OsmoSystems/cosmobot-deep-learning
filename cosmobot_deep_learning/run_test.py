import os
from unittest.mock import sentinel

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


@pytest.fixture
def mock_get_gpu_stats(mocker):
    return mocker.patch.object(module, "_get_gpu_stats")


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

        hyperparameters = {"dataset_cache_name": "test"}

        actual_dataset = module._prepare_dataset_with_caching(
            prepare_dataset=sentinel.prepare_dataset, hyperparameters=hyperparameters
        )

        mock_dataset_cache_helpers["_get_prepared_dataset"].assert_called_once_with(
            prepare_dataset=sentinel.prepare_dataset, hyperparameters=hyperparameters
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

        hyperparameters = {"dataset_cache_name": "test"}

        actual_dataset = module._prepare_dataset_with_caching(
            prepare_dataset=sentinel.prepare_dataset, hyperparameters=hyperparameters
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

        hyperparameters = {"dataset_cache_name": None}

        actual_dataset = module._prepare_dataset_with_caching(
            prepare_dataset=sentinel.prepare_dataset, hyperparameters=hyperparameters
        )

        mock_dataset_cache_helpers["_get_prepared_dataset"].assert_called_once_with(
            prepare_dataset=sentinel.prepare_dataset, hyperparameters=hyperparameters
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


class TestSetCUDAVisibleDevices:
    def test_sets_cpu_mode_on_dryrun(self):
        module._set_cuda_visible_devices(no_gpu=False, dryrun=True)

        assert os.getenv("CUDA_VISIBLE_DEVICES") == "-1"

    def test_sets_cpu_mode_when_gpu_disabled(self):
        module._set_cuda_visible_devices(no_gpu=True, dryrun=False)

        assert os.getenv("CUDA_VISIBLE_DEVICES") == "-1"

    @pytest.mark.parametrize(
        "stats,expected",
        (
            ({"GPU ID": [0], "Memory Free (MiB)": [7001]}, "0"),
            ({"GPU ID": [0, 1], "Memory Free (MiB)": [7001, 10]}, "0"),
            ({"GPU ID": [0, 1], "Memory Free (MiB)": [10, 7001]}, "1"),
            ({"GPU ID": [0, 1, 2], "Memory Free (MiB)": [10, 10, 7001]}, "2"),
            ({"GPU ID": [0, 1, 2, 3], "Memory Free (MiB)": [10, 10, 10, 7001]}, "3"),
        ),
    )
    def test_gets_first_available_gpu(self, mock_get_gpu_stats, stats, expected):
        test_stats = pd.DataFrame(stats)
        mock_get_gpu_stats.return_value = test_stats

        module._set_cuda_visible_devices(no_gpu=False, dryrun=False)

        assert os.getenv("CUDA_VISIBLE_DEVICES") == expected

    def test_raises_if_no_available_gpu(self, mock_get_gpu_stats):
        test_stats = pd.DataFrame({"GPU ID": [0], "Memory Free (MiB)": [6999]})
        mock_get_gpu_stats.return_value = test_stats

        with pytest.raises(module.NoGPUAvailable):
            module._set_cuda_visible_devices(no_gpu=False, dryrun=False)

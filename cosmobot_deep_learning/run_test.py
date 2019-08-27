from unittest.mock import Mock, sentinel

import pytest
import pandas as pd

from . import run as module


@pytest.fixture
def mock_get_dataset_cache_filepath(mocker):
    return mocker.patch.object(module, "get_dataset_cache_filepath")


@pytest.fixture
def mock_dataset_cache_helpers(mocker):
    mock_load_dataset_cache = mocker.patch.object(module, "_load_dataset_cache")
    mock_save_dataset_cache = mocker.patch.object(module, "_save_dataset_cache")
    return {
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
    test_dataset = ("1", "2", "3", "4")

    def test_saves_new_dataset_cache_when_file_doesnt_exist(
        self, mocker, mock_get_dataset_cache_filepath, mock_dataset_cache_helpers
    ):
        mocker.patch("os.path.isfile", return_value=False)
        mock_prepare_dataset = Mock(return_value=self.test_dataset)
        mock_get_dataset_cache_filepath.return_value = sentinel.cache_filepath

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=sentinel.raw_datatset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dataset_cache_name="test",
        )

        mock_prepare_dataset.assert_called_once_with(
            raw_dataset=sentinel.raw_datatset, hyperparameters=sentinel.hyperparameters
        )
        assert actual_dataset == self.test_dataset
        mock_dataset_cache_helpers["_load_dataset_cache"].assert_not_called()
        mock_dataset_cache_helpers["_save_dataset_cache"].assert_called_once_with(
            sentinel.cache_filepath, self.test_dataset
        )

    def test_loads_existing_dataset_cache_when_file_exists(
        self, mocker, mock_get_dataset_cache_filepath, mock_dataset_cache_helpers
    ):
        mocker.patch("os.path.isfile", return_value=True)
        mock_prepare_dataset = Mock()
        mock_get_dataset_cache_filepath.return_value = sentinel.cache_filepath

        mock_dataset_cache_helpers[
            "_load_dataset_cache"
        ].return_value = sentinel.mock_dataset

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=sentinel.raw_datatset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dataset_cache_name="test",
        )

        mock_prepare_dataset.assert_not_called()
        mock_dataset_cache_helpers["_load_dataset_cache"].assert_called_once_with(
            sentinel.cache_filepath
        )
        mock_dataset_cache_helpers["_save_dataset_cache"].assert_not_called()
        assert actual_dataset == sentinel.mock_dataset

    def test_ignores_cache_when_no_name_specified(
        self, mocker, mock_get_dataset_cache_filepath, mock_dataset_cache_helpers
    ):
        mock_prepare_dataset = Mock(return_value=self.test_dataset)

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=sentinel.raw_datatset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=sentinel.hyperparameters,
            dataset_cache_name=None,
        )

        mock_prepare_dataset.assert_called_once_with(
            raw_dataset=sentinel.raw_datatset, hyperparameters=sentinel.hyperparameters
        )
        mock_dataset_cache_helpers["_load_dataset_cache"].assert_not_called()
        mock_dataset_cache_helpers["_save_dataset_cache"].assert_not_called()
        assert actual_dataset == self.test_dataset

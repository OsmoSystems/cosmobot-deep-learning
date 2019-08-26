import os
import pickle
from unittest.mock import Mock, sentinel

import pytest
import pandas as pd

from . import run as module


@pytest.fixture
def tmp_cache_filepath(tmp_path):
    return tmp_path / "test.pickle"


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
    def test_saves_new_dataset_cache(self, tmp_cache_filepath):
        test_dataset = "serialize me"
        mock_raw_dataset = pd.DataFrame([{"test": "value"}])
        mock_hyperparameters = {
            "dataset_cache_name": "test",
            "use_cache": True,
            "dataset_cache_filepath": tmp_cache_filepath,
        }

        mock_prepare_dataset = Mock(return_value=test_dataset)

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=mock_raw_dataset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=mock_hyperparameters,
        )

        mock_prepare_dataset.assert_called_once_with(
            raw_dataset=mock_raw_dataset, hyperparameters=mock_hyperparameters
        )
        assert actual_dataset == test_dataset
        assert os.path.isfile(tmp_cache_filepath)

        with open(tmp_cache_filepath, "rb") as f:
            # Pickling doesn't preserve sentinel identity, so check the name
            assert pickle.load(f) == test_dataset

    def test_loads_existing_dataset_cache(self, tmp_cache_filepath):
        test_dataset = "serialize me"
        mock_hyperparameters = {
            "dataset_cache_name": "test",
            "use_cache": True,
            "dataset_cache_filepath": tmp_cache_filepath,
        }
        mock_prepare_dataset = Mock(return_value=test_dataset)

        with open(tmp_cache_filepath, "wb+") as f:
            pickle.dump(test_dataset, f)

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=pd.DataFrame(),
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=mock_hyperparameters,
        )

        mock_prepare_dataset.assert_not_called()
        assert actual_dataset == test_dataset

    def test_ignores_cache_when_not_specified(self, mocker):
        mock_open = mocker.patch.object(module, "open")
        mock_raw_dataset = pd.DataFrame([{"test": "value"}])
        mock_hyperparameters = {
            "dataset_cache_name": None,
            "use_cache": False,
            "dataset_cache_filepath": None,
        }

        mock_prepare_dataset = Mock(return_value=sentinel.dataset)

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=mock_raw_dataset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=mock_hyperparameters,
        )

        mock_prepare_dataset.assert_called_once_with(
            raw_dataset=mock_raw_dataset, hyperparameters=mock_hyperparameters
        )
        mock_open.assert_not_called()
        assert actual_dataset == sentinel.dataset

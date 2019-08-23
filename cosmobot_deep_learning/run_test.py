import os
import pickle
from unittest.mock import Mock, sentinel

import pytest
import pandas as pd

from . import run as module


@pytest.fixture
def mock_prepare_dataset():
    return Mock(return_value=sentinel.dataset)


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
    def test_saves_new_dataset_cache(self, tmp_path, mock_prepare_dataset):
        mock_raw_dataset = pd.DataFrame([{"test": "value"}])
        mock_hyperparameters = {"test": "value"}

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=mock_raw_dataset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=mock_hyperparameters,
            dataset_cache_name="test",
            cache_directory=tmp_path,
        )

        expected_output_file = os.path.join(tmp_path, "test.pickle")

        mock_prepare_dataset.assert_called_once_with(
            raw_dataset=mock_raw_dataset, hyperparameters=mock_hyperparameters
        )
        assert actual_dataset == sentinel.dataset
        assert os.path.isfile(expected_output_file)

        with open(expected_output_file, "rb") as f:
            # Pickling doesn't preserve sentinel identity, so check the name
            assert pickle.load(f).name == sentinel.dataset.name

    def test_loads_existing_dataset_cache(self, tmp_path, mock_prepare_dataset):
        output_file = os.path.join(tmp_path, "test.pickle")

        with open(output_file, "wb+") as f:
            pickle.dump(sentinel.dataset, f)

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=pd.DataFrame(),
            prepare_dataset=mock_prepare_dataset,
            hyperparameters={},
            dataset_cache_name="test",
            cache_directory=tmp_path,
        )

        mock_prepare_dataset.assert_not_called()
        assert actual_dataset.name == sentinel.dataset.name

    def test_ignores_cache_when_not_specified(
        self, mocker, tmp_path, mock_prepare_dataset
    ):
        mock_open = mocker.patch.object(module, "open")
        mock_raw_dataset = pd.DataFrame([{"test": "value"}])
        mock_hyperparameters = {"test": "value"}

        actual_dataset = module._prepare_dataset_with_caching(
            raw_dataset=mock_raw_dataset,
            prepare_dataset=mock_prepare_dataset,
            hyperparameters=mock_hyperparameters,
            dataset_cache_name=None,
            cache_directory=tmp_path,
        )

        mock_prepare_dataset.assert_called_once_with(
            raw_dataset=mock_raw_dataset, hyperparameters=mock_hyperparameters
        )
        mock_open.assert_not_called()
        assert actual_dataset == sentinel.dataset

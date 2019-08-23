from unittest.mock import sentinel

import pandas as pd

from . import run as module


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

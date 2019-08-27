from unittest.mock import sentinel

import pandas as pd
from pandas.util.testing import assert_frame_equal
import pytest

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

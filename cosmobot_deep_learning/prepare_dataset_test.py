import pandas as pd
import numpy as np

from . import prepare_dataset as module


class TestExtractInputs:
    def test_returns_dataset_slimmed_to_columns(self):
        raw_dataset = pd.DataFrame(
            {
                "column_to_include": [1, 2, 3],
                "other_column": [4, 5, 6],
                "another_column": [7, 8, 9],
                "DO patch r_msorm": [4, 5, 6],
                "reference patch r_msorm": [7, 8, 9],
            }
        )

        expected = np.array([[1, 2, 3]]).T

        actual = module.extract_inputs(
            raw_dataset, input_column_names=["column_to_include"]
        )

        np.testing.assert_array_equal(actual, expected)

    def test_returns_dataset_with_sr_column(self):
        raw_dataset = pd.DataFrame(
            {
                "column_to_include": [1, 2, 3],
                "DO patch r_msorm": [4, 5, 6],
                "reference patch r_msorm": [7, 8, 9],
            }
        )

        expected = np.array(
            [
                # fmt: off
                [1, 2, 3],
                [4 / 7, 5 / 8, 6 / 9]
                # fmt: on
            ]
        ).T

        actual = module.extract_inputs(
            raw_dataset, input_column_names=["column_to_include", "sr"]
        )

        np.testing.assert_array_equal(actual, expected)


class TestExtractLabels:
    # TODO
    pass


class TestPrepareDatasetNumerical:
    # TODO
    pass

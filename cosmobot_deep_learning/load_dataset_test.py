import os
import pkg_resources

import pandas as pd
import pytest

from . import load_dataset as module
from . import s3 as s3_module


class ComparableSeries(pd.Series):
    """ pandas Series patched to allow equality testing """

    def __eq__(self, other):
        return (super(ComparableSeries, self).values == other.values).all()


@pytest.fixture
def mock_download_s3_files(mocker):
    return mocker.patch.object(s3_module, "_download_s3_files")


class TestLoadMultiExperimentDatasetCsv:
    def test_downloads_files_to_correct_local_paths_and_returns_dataframe(
        self, tmp_path, mocker, mock_download_s3_files
    ):
        test_df = pd.DataFrame(
            [
                {
                    "experiment": "experiment_1",
                    "image": "image_1.jpeg",
                    "other": "other",
                },
                {
                    "experiment": "experiment_2",
                    "image": "image_2.jpeg",
                    "other": "other",
                },
                {
                    "experiment": "experiment_2",
                    "image": "image_3.jpeg",
                    "other": "other",
                },
            ]
        )
        csv_filepath = os.path.join(tmp_path, "big_special_dataset.csv")
        test_df.to_csv(csv_filepath, index=False)

        expected_df = test_df.copy()
        expected_df["local_filepath"] = [
            os.path.join(module.LOCAL_DATA_DIRECTORY, "experiment_1", "image_1.jpeg"),
            os.path.join(module.LOCAL_DATA_DIRECTORY, "experiment_2", "image_2.jpeg"),
            os.path.join(module.LOCAL_DATA_DIRECTORY, "experiment_2", "image_3.jpeg"),
        ]

        actual_df = module.load_multi_experiment_dataset_csv(csv_filepath)

        pd.testing.assert_frame_equal(expected_df, actual_df)

    # This is a regression test for a strange issue where progress_apply sometimes
    # returns a series of the wrong shape, e.g. (1, 10) instead of (10,).
    # I seem to get a consistent repro when there is only one experiment
    def test_loads_single_experiment_dataset_files(
        self, tmp_path, mocker, mock_download_s3_files
    ):

        test_df = pd.DataFrame(
            [
                {"experiment": "experiment_1", "image": "image_1.jpeg"},
                {"experiment": "experiment_1", "image": "image_2.jpeg"},
            ]
        )
        csv_filepath = os.path.join(tmp_path, "big_special_dataset.csv")
        test_df.to_csv(csv_filepath, index=False)

        # Test is just that this doesn't blow up
        # Error looks like: "ValueError: Wrong number of items passed 2, placement implies 1"
        module.load_multi_experiment_dataset_csv(csv_filepath)


class TestGetPkgDatasetFilepath:
    def test_returns_correct_path_within_repo(self):
        actual = module.get_pkg_dataset_filepath("validation_dataset.csv")

        expected = pkg_resources.resource_filename(
            "cosmobot_deep_learning", "datasets/validation_dataset.csv"
        )

        assert actual == expected


class TestGetDatasetHash:
    def test_returns_correct_hash(self):
        dataset_filepath = pkg_resources.resource_filename(
            "cosmobot_deep_learning", "datasets/validation_dataset.csv"
        )
        actual = module.get_dataset_hash(dataset_filepath)
        assert actual == "96d36585167260174642ff00a646f797"

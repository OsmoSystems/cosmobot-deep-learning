import os

import pandas as pd

from . import load_dataset as module


class ComparableSeries(pd.Series):
    """ pandas Series patched to allow equality testing """

    def __eq__(self, other):
        return (super(ComparableSeries, self) == other).all()


class TestLoadMultiExperimentDatasetCsv:
    def test_calls_get_local_filepaths_with_appropriate_info_and_returns_dataframe(
        self, tmp_path, mocker
    ):
        filename = "filenameeeee.jpeg"
        experiment_name = "expeeeeeriment"
        test_df = pd.DataFrame(
            [{"experiment": experiment_name, "image": filename, "other": "other"}]
        )
        csv_filename = "big_special_dataset.csv"
        csv_filepath = os.path.join(tmp_path, csv_filename)
        test_df.to_csv(csv_filepath, index=False)

        local_jpeg_path = mocker.sentinel.local_jpeg_path
        expected_returned_dataframe = test_df.copy()
        expected_returned_dataframe["local_filepath"] = [local_jpeg_path]

        mock_download_s3_files_and_get_local_filepaths = mocker.patch.object(
            module, "naive_sync_from_s3", return_value=pd.Series([local_jpeg_path])
        )

        actual_returned_dataframe = module.load_multi_experiment_dataset_csv(
            csv_filepath
        )

        mock_download_s3_files_and_get_local_filepaths.assert_called_with(
            experiment_directory=experiment_name,
            file_names=ComparableSeries([filename]),
            output_directory_path=os.path.expanduser(
                "~/osmo/cosmobot-data-sets/big_special_dataset"
            ),
        )

        pd.testing.assert_frame_equal(
            expected_returned_dataframe, actual_returned_dataframe
        )

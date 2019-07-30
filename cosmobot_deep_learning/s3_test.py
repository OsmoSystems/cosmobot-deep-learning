import os

import pandas as pd
import pytest

from . import s3 as module


@pytest.fixture
def mock_download_s3_files(mocker):
    return mocker.patch.object(module, "_download_s3_files")


@pytest.fixture
def mock_check_call(mocker):
    mock_check_call = mocker.patch.object(module, "check_call")
    mock_check_call.return_value = None

    return mock_check_call


# COPY-PASTA: This was copied from cosmobot-process-experiment
class TestDownloadS3Files:
    def test_calls_s3_sync_command(self, mock_check_call):
        module._download_s3_files(
            experiment_directory="my_experiment",
            file_names=["image1.jpeg", "image2.jpeg"],
            output_directory_path="local_sync_path",
        )

        expected_command = (
            "aws s3 sync s3://camera-sensor-experiments/my_experiment local_sync_path "
            '--exclude "*" '
            '--include "image1.jpeg" --include "image2.jpeg"'
        )

        mock_check_call.assert_called_with([expected_command], shell=True)

    def test_many_images_batched_properly(self, mock_check_call):
        batch_size = 30
        test_file_names = [f"test_image{i}.jpeg" for i in range(batch_size + 1)]

        module._download_s3_files(
            experiment_directory="my_experiment",
            file_names=test_file_names,
            output_directory_path="local_sync_path",
        )

        # Chained indexing to first expected command:
        # First call() in call list, arguments provided to first call (tuple),
        # first argument provided (list), first item in list (command string)
        first_expected_command = mock_check_call.call_args_list[0][0][0][0]

        second_expected_command = (
            "aws s3 sync s3://camera-sensor-experiments/my_experiment local_sync_path "
            '--exclude "*" '
            f'--include "test_image{batch_size}.jpeg"'
        )

        assert mock_check_call.call_count == 2
        assert "test_image0.jpeg" in first_expected_command
        assert f"test_image{batch_size - 1}.jpeg" in first_expected_command
        assert f"test_image{batch_size}.jpeg" not in first_expected_command
        assert first_expected_command.count("test_image") == 30
        mock_check_call.assert_called_with([second_expected_command], shell=True)


class TestNaiveSyncFromS3:
    def test_returns_filepaths_series(self, mock_download_s3_files):
        actual_local_filepaths = module.naive_sync_from_s3(
            experiment_directory="experiment_dir",
            file_names=pd.Series(["filename_1", "filename_2"]),
            output_directory_path="local_dir",
        )

        expected_local_filepaths = pd.Series(
            [
                os.path.join("local_dir", "filename_1"),
                os.path.join("local_dir", "filename_2"),
            ]
        )
        pd.testing.assert_series_equal(actual_local_filepaths, expected_local_filepaths)

    def test_skips_sync_when_all_files_present(self, mocker, mock_download_s3_files):
        mocker.patch.object(module.os.path, "isfile", return_value=True)

        module.naive_sync_from_s3(
            experiment_directory="experiment_dir",
            file_names=pd.Series(["filename_1", "filename_2"]),
            output_directory_path="local_dir",
        )

        mock_download_s3_files.assert_not_called()

    def test_performs_sync_when_any_file_not_present(
        self, mocker, mock_download_s3_files
    ):
        mocker.patch.object(module.os.path, "isfile", side_effect=[False, True])

        experiment_directory = "experiment_dir"
        file_names = pd.Series(["filename_1", "filename_2"])
        output_directory_path = "local_dir"

        module.naive_sync_from_s3(
            experiment_directory=experiment_directory,
            file_names=file_names,
            output_directory_path=output_directory_path,
        )

        mock_download_s3_files.assert_called_with(
            experiment_directory, file_names, output_directory_path
        )

    def test_does_reasonable_things_when_no_files_passed(self, mock_download_s3_files):
        expected_local_filepaths = pd.Series([])

        actual_local_filepaths = module.naive_sync_from_s3(
            experiment_directory="experiment_dir",
            file_names=pd.Series([]),
            output_directory_path="local_dir",
        )

        mock_download_s3_files.assert_not_called()
        pd.testing.assert_series_equal(actual_local_filepaths, expected_local_filepaths)

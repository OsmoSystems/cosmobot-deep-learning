import os

from subprocess import check_call
from typing import List

import pandas as pd


CAMERA_SENSOR_EXPERIMENTS_BUCKET_NAME = "camera-sensor-experiments"


# COPY-PASTA: This was copied from cosmobot-process-experiment
def _download_s3_files(
    experiment_directory: str, file_names: List[str], output_directory_path: str
) -> None:
    """ Download specific filenames from within an experiment directory on s3.
    Suitable for syncing files that change over time - uses s3 sync's timestamp-based approach to checking if a
    particular file needs to be re-synced.
    """

    # Our implementation for performing a filtered download of files from s3 chokes when attempting to
    # download large numbers of files. To avoid this problem, perform the download in batches.
    # 30 appears to be a safe batch-size limit
    batch_size = 30
    file_name_batches = file_name_batches = [
        file_names[batch_start_index : batch_start_index + batch_size]
        for batch_start_index in range(0, len(file_names), batch_size)
    ]

    for batch_file_names in file_name_batches:

        include_args = " ".join(
            [f'--include "{file_name}"' for file_name in batch_file_names]
        )

        # Would be better to use boto, but neither boto nor boto3 support sync
        # https://github.com/boto/boto3/issues/358
        command = (
            f"aws s3 sync s3://{CAMERA_SENSOR_EXPERIMENTS_BUCKET_NAME}/{experiment_directory} {output_directory_path} "
            f'--exclude "*" {include_args}'
        )
        check_call([command], shell=True)


def naive_sync_from_s3(
    experiment_directory: str, file_names: pd.Series, output_directory_path: str
) -> pd.Series:
    """ Sync a Series of files from s3, returning local file paths corresponding to the synced files.
    Uses a naive approach to checking if sync is necessary - if all s3 filenames are present locally, skips syncing.
    This makes this function unsuitable for syncing files that change over time.
    """
    local_filepaths = file_names.apply(
        lambda filename: os.path.join(output_directory_path, filename)
    )
    already_downloaded = local_filepaths.apply(os.path.isfile)
    if not already_downloaded.all():
        _download_s3_files(experiment_directory, file_names, output_directory_path)

    return local_filepaths

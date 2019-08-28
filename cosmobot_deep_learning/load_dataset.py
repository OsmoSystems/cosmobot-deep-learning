import hashlib
import os
import pkg_resources

import pandas as pd
from tqdm.auto import tqdm

from .s3 import naive_sync_from_s3


PACKAGE_NAME = "cosmobot_deep_learning"
LOCAL_DATA_DIRECTORY = os.path.expanduser("~/osmo/cosmobot-data-sets/")
LOCAL_CACHE_DIRECTORY = os.path.expanduser("~/osmo/cosmobot-dataset-cache/")


def _get_files_for_experiment_df(experiment_df_group):
    # The experiment_df_group is a groupby object which has a .name property
    # corresponding to the groupby value (in this case the experiment name)
    experiment_directory = experiment_df_group.name

    image_filenames = experiment_df_group["image"]

    local_image_files_directory = os.path.join(
        LOCAL_DATA_DIRECTORY, experiment_directory
    )

    return naive_sync_from_s3(
        experiment_directory=experiment_directory,
        file_names=image_filenames,
        output_directory_path=local_image_files_directory,
    )


def get_dataset_with_local_filepaths(dataset: pd.DataFrame) -> pd.DataFrame:
    """ For a pre-prepared ML dataset, load the DataFrame with local image paths, optionally downloading said images
    Note that syncing tends to take a long time, though syncing for individual experiments will be skipped if all files
    are already downloaded.

    Args:
        dataset: ML dataset DataFrame. DataFrame is expected to have at least the following columns:
            'experiment': experiment directory on s3
            'image': image filename on s3
            All other columns are passed through.

    Returns:
        The dataset provided with the additional column 'local_filepath' which will contain file paths of
        the locally stored images.

    Side-effects:
        * syncs images corresponding to the ML dataset from s3 to their associated experiment directory:
            ~/osmo/cosmobot-data-sets/{experiment_directory}/
        * prints status messages so that the user can keep track of this very slow operation
        * calls tqdm.auto.tqdm.pandas() which patches pandas datatypes to have `.progress_apply()` methods
    """
    # Side effect: patch pandas datatypes to have .progress_apply() methods
    tqdm.pandas()

    # Group by experiment so that we can download from each experiment folder on s3
    dataset_by_experiment = dataset.groupby(
        "experiment", as_index=False, group_keys=False
    )

    print(
        "This can be a *very* slow, uneven progress bar PLUS it is off by one or something so please wait until I tell "
        "you I am done:"
    )

    local_filepaths = dataset_by_experiment.progress_apply(_get_files_for_experiment_df)

    print("Done syncing images. thanks for waiting.")

    # Transpose because progress_apply on the groupby object return series of the wrong shape
    # e.g. (1, 10) instead of (10,), when there's only one experiment. When there are multiple
    # experiments, it returns the correct single-dimensional shape, and transpose has no effect
    dataset["local_filepath"] = local_filepaths.T
    return dataset


def get_dataset_cache_filepath(dataset_cache_name):
    """ Returns the filepath for a named dataset cache to be loaded or created.
    """
    return os.path.join(LOCAL_CACHE_DIRECTORY, f"{dataset_cache_name}.pickle")


def get_pkg_dataset_filepath(dataset_filename):
    """ Returns the filepath to the given dataset inside this package.
        (Assumes the dataset is correctly checked into this package.)
    """
    dataset_filepath = pkg_resources.resource_filename(
        PACKAGE_NAME, f"datasets/{dataset_filename}"
    )
    return dataset_filepath


def get_dataset_hash(dataset_filepath):
    with open(dataset_filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

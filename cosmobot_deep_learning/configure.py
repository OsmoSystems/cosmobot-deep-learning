import argparse
from pathlib import Path

from typing import List


def parse_model_run_args(args: List[str]) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.RawTextHelpFormatter
    )

    arg_parser.add_argument(
        "--gpu",
        # required=True,
        type=int,
        help=(
            "Select GPU for training by CUDA Device ID number (e.g. 0 - 3).\n"
            "Run `nvidia-smi` to see available devices and IDs.\n"
            "Use -1 to disable GPU taining."
        ),
        dest="gpu",
    )

    arg_parser.add_argument(
        "--dryrun",
        required=False,
        action="store_true",
        default=False,
        help=(
            "Perform a dry run with a tiny dataset to check that a model will compile."
        ),
    )

    arg_parser.add_argument(
        "--dataset-cache",
        required=False,
        type=str,
        dest="dataset_cache_name",
        metavar="CACHED-DATASET-NAME",
        help=(
            "Name of cached dataset to load, or, if named dataset doesn't exist, save "
            "the prepared dataset created during this run with the provided name.\n"
            "NOTE that if the named dataset cache does exist, all preprocessing and preparation will "
            "be skipped on this run. "
            "Files are saved in ~/osmo/cosmobot-dataset-cache/ with a .pickle extension."
        ),
    )

    arg_parser.add_argument(
        "--desired-train-sample-count", type=int, dest="desired_train_sample_count"
    )

    arg_namespace = arg_parser.parse_args(args)
    return arg_namespace


def get_model_name_from_filepath(filepath: str) -> str:
    """ Given the filepath for the model file, extracts a friendlier model name:
        The filename minus extension

    Example:
        >>> get_model_name_from_filepath("/path/to/my_model.py")
        "my_model"
    """
    return Path(filepath).stem

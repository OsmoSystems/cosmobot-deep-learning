import argparse
from pathlib import Path

from typing import List

from cosmobot_deep_learning.constants import OPTIMIZER_CLASSES_BY_NAME


def _string_to_bool(v):
    """This is used to allow an argument to be set to a true/false string value.
    We are using this to support W&B's hyperparameter sweep agent, which will pass boolean values
    in as --some-flag=True or --some-flag=False.

    To use it on a boolean argument, set the following properties set in parser.add_argument():
        required=False,
        type=_string_to_bool,
        nargs="?",
        const=True,
        default=False,

    This function copied and modified from https://stackoverflow.com/a/43357954
    """
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean string or boolean expected; received {v}."
        )


def parse_model_run_args(
    args: List[str], model_hyperparameter_parser: argparse.ArgumentParser = None
) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[model_hyperparameter_parser] if model_hyperparameter_parser else [],
    )

    arg_parser.add_argument(
        "--no-gpu",
        required=False,
        type=_string_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=("Disable GPU for training."),
        dest="no_gpu",
    )

    # --dryrun=True is allowed so that hyperparameter sweeps can use it
    arg_parser.add_argument(
        "--dryrun",
        required=False,
        type=_string_to_bool,
        nargs="?",
        const=True,
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

    arg_parser.add_argument("--epochs", type=int)

    arg_parser.add_argument(
        "--optimizer-name", choices=list(OPTIMIZER_CLASSES_BY_NAME.keys())
    )
    arg_parser.add_argument("--learning-rate", type=float)

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

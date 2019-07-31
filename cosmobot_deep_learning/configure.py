import argparse

from typing import List, Dict


DEFAULT_EPOCHS = 10000
DEFAULT_BATCH_SIZE = 125


def parse_args(args: List[str]) -> Dict:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="Value to be passed to CUDA_VISIBLE_DEVICES environment variable",
        dest="gpu",
    )

    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"number of epochs for training (default: {DEFAULT_EPOCHS})",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"input batch size for training (default: {DEFAULT_BATCH_SIZE})",
    )

    arg_namespace = arg_parser.parse_args(args)
    return vars(arg_namespace)

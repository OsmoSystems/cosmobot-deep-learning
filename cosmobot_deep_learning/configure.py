import argparse

from typing import List


DEFAULT_EPOCHS = 10000
DEFAULT_BATCH_SIZE = 125


def parse_model_run_args(args: List[str]) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.RawTextHelpFormatter
    )

    arg_parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help=(
            "Select GPU for training by CUDA Device ID number (e.g. 0 - 3).\n"
            "Run `nvidia-smi` to see available devices and IDs.\n"
            "Use -1 to disable GPU taining."
        ),
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
    return arg_namespace

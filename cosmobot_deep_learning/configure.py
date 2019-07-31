import argparse

from typing import List, Dict


def parse_args(args: List[str]) -> Dict:
    arg_parser = argparse.ArgumentParser(
        description=(""), formatter_class=argparse.RawTextHelpFormatter
    )

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
        default=10000,
        help="number of epochs for training (default: 10000)",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=125,
        help="input batch size for training (default: 125)",
    )

    arg_namespace = arg_parser.parse_args(args)
    return vars(arg_namespace)

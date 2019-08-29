"""
This model is a dense-layer network that trains only on two numeric inputs:
 - temperature
 - spatial ratiometric ("OO DO patch Wet r_msorm" / "Type 1 Chemistry Hand Applied Dry r_msorm")
"""
import os
import sys

import keras
import tensorflow as tf

from cosmobot_deep_learning.configure import (
    parse_model_run_args,
    get_model_name_from_filepath,
)
from cosmobot_deep_learning.hyperparameters import get_hyperparameters
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_numeric
from cosmobot_deep_learning.run import run


DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "dataset_filename": "2019-08-09--14-33-26_osmo_ml_dataset.csv",
    "batch_size": 3000,
    "numeric_input_columns": ["sr", "PicoLog temperature (C)"],
}


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    x_train_samples_count, numeric_inputs_count = x_train.shape

    sr_model = keras.models.Sequential(
        [
            keras.layers.Dense(
                11, activation=tf.nn.relu, input_shape=[numeric_inputs_count]
            ),
            keras.layers.Dense(32),
            keras.layers.advanced_activations.LeakyReLU(),
            keras.layers.Dense(1, name="sv_DO"),
        ]
    )

    sr_model.compile(
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return sr_model


def _set_cuda_visible_devices(hyperparameters):
    gpu_arg = hyperparameters.get("gpu")

    if gpu_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_arg)
    else:
        # for sweeps, CUDA_VISIBLE_DEVICES should be set to the desired gpu when running each agent, example:
        # CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id>
        # CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id>

        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        hyperparameters["gpu"] = gpu

        if gpu is None:
            # TODO use better exception class. should we also validate the value? scope creep?
            raise Exception(
                "Must specify --gpu or have CUDA_VISIBLE_DEVICES set in environment"
            )


def main(command_line_args):
    args = parse_model_run_args(command_line_args)

    # for sweeps, this should be set when running the agent(s), example:
    # CUDA_VISIBLE_DEVICES=0 wandb agent mcg70107
    # CUDA_VISIBLE_DEVICES=1 wandb agent mcg70107
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # HACK HACK HACK
    hyperparameters_from_args = {k: v for k, v in vars(args).items() if v is not None}

    # TODO hack (we might not want all args in hyperparameters)
    hyperparameters = get_hyperparameters(
        **{**DEFAULT_HYPERPARAMETERS, **hyperparameters_from_args}
    )

    # TODO remove
    print(hyperparameters)

    run(
        hyperparameters,
        prepare_dataset_numeric,
        create_model,
        dryrun=args.dryrun,
        dataset_cache_name=args.dataset_cache,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

"""
This model is a dense-layer network that trains only on two numerical inputs:
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
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_numerical
from cosmobot_deep_learning.run import run


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    input_numerical_data_dimensions = x_train.shape[1]
    sr_model = keras.models.Sequential(
        [
            keras.layers.Dense(
                11, activation=tf.nn.relu, input_shape=[input_numerical_data_dimensions]
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


if __name__ == "__main__":
    args = parse_model_run_args(sys.argv[1:])

    # Note: we may eventually need to change how we set this to be compatible with
    # hyperparameter sweeps. See https://www.wandb.com/articles/multi-gpu-sweeps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # TODO: remove
    os.environ["WANDB_MODE"] = "dryrun"

    hyperparameters = get_hyperparameters(
        model_name=get_model_name_from_filepath(__file__),
        # TODO: revert these for-testing changes
        dataset_filename="2019-08-09--14-33-26_osmo_ml_dataset_tiny.csv",
        epochs=1,
        # End TODO
        batch_size=3000,
        input_columns=["sr", "PicoLog temperature (C)"],
    )

    run(hyperparameters, prepare_dataset_numerical, create_model)

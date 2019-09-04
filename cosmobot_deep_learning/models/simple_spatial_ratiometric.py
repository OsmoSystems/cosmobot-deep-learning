"""
This model is a dense-layer network that trains only on two numeric inputs:
 - temperature
 - spatial ratiometric ("OO DO patch Wet r_msorm" / "Type 1 Chemistry Hand Applied Dry r_msorm")
"""
import sys

import keras
import tensorflow as tf

from cosmobot_deep_learning.configure import get_model_name_from_filepath
from cosmobot_deep_learning.hyperparameters import (
    get_hyperparameters_from_args,
    get_optimizer,
)
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_numeric
from cosmobot_deep_learning.run import run


DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "batch_size": 3000,
    "numeric_input_columns": ["sr", "PicoLog temperature (C)"],
}


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    optimizer = get_optimizer(hyperparameters)

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
        optimizer=optimizer,
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return sr_model


def main(command_line_args):
    hyperparameters = get_hyperparameters_from_args(
        command_line_args, DEFAULT_HYPERPARAMETERS
    )

    run(hyperparameters, prepare_dataset_numeric, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

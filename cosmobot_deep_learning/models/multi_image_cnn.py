"""
This model is an N-branch network that combines:
1. A configurable number of hand-made CNNs with 3 convolutional layers each that train on distinct ROI inputs
2. A dense network that trains on numeric inputs:
    - temperature
    - numeric output of each of the hand-made CNNs
"""

import argparse
import sys

import keras
from keras import regularizers
from keras_drop_block import DropBlock2D

from cosmobot_deep_learning.configure import get_model_name_from_filepath
from cosmobot_deep_learning.hyperparameters import (
    get_hyperparameters_from_args,
    get_optimizer,
)
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_ROIs_and_numeric
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)

DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "numeric_input_columns": ["PicoLog temperature (C)"],
    "image_size": 32,
    # ROI names to extract from `ROI definitions` column in the dataset.
    # WARNING: The order here is preserved through data processing and model creation / input
    # If you are using a cached dataset, make sure you have the correct order.
    "input_ROI_names": [
        "DO patch",
        "reference patch",
        "reflectance standard",
        "center",
    ],
    "kernel_initializer": keras.initializers.he_normal(),
    "kernel_size": (4, 4),
    "dense_layer_units": 128,
    "dropblock_size": 5,
    "dropblock_keep_prob": 0.9,
}


def get_convolutional_input(branch_id, x_train_sample_image, hyperparameters):
    # Layer names cannot have spaces
    model_branch_id = branch_id.replace(" ", "_")

    block_size = hyperparameters["dropblock_size"]
    keep_prob = hyperparameters["dropblock_keep_prob"]
    convolutional_kernel_size = hyperparameters["convolutional_kernel_size"]

    shared_layer_kwargs = {
        "kernel_size": (convolutional_kernel_size, convolutional_kernel_size),
        "activation": "relu",
        "kernel_initializer": hyperparameters["kernel_initializer"],
    }

    convolutional_sub_model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                16, input_shape=x_train_sample_image.shape, **shared_layer_kwargs
            ),
            keras.layers.MaxPooling2D(2),
            DropBlock2D(
                block_size=block_size,
                keep_prob=keep_prob,
                name=f"{model_branch_id}-drop-block-1",
            ),
            keras.layers.Conv2D(32, **shared_layer_kwargs),
            keras.layers.MaxPooling2D(2),
            DropBlock2D(
                block_size=block_size,
                keep_prob=keep_prob,
                name=f"{model_branch_id}-drop-block-2",
            ),
            keras.layers.Conv2D(32, **shared_layer_kwargs),
            keras.layers.Flatten(name=f"{model_branch_id}-prep-for-dense"),
            keras.layers.Dense(64, **shared_layer_kwargs),
            keras.layers.Dense(
                64,
                name=f"{model_branch_id}-final_dense",
                kernel_initializer=hyperparameters["kernel_initializer"],
            ),
        ]
    )

    return convolutional_sub_model


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    input_ROI_names = hyperparameters["input_ROI_names"]
    optimizer = get_optimizer(hyperparameters)

    # x_train is a list of [numeric_inputs, ROI_input_1, ..., ROI_input_x]
    assert len(input_ROI_names) == len(x_train) - 1
    x_train_numeric = x_train[0]
    x_train_samples_count, numeric_inputs_count = x_train_numeric.shape

    ROI_inputs = [
        get_convolutional_input(ROI_name, x_train[i + 1][0], hyperparameters)
        for i, ROI_name in enumerate(input_ROI_names)
    ]

    temperature_input = keras.layers.Input(
        shape=(numeric_inputs_count,), name="temperature"
    )

    kernel_initializer = hyperparameters["kernel_initializer"]
    dense_layer_units = hyperparameters["dense_layer_units"]
    dense_layer_kwargs = {
        "units ": dense_layer_units,
        "activation": "relu",
        "kernel_initializer": kernel_initializer,
        "kernel_regularizer": regularizers.l2(0.01),
    }

    temp_and_image_add = keras.layers.concatenate(
        [temperature_input] + [ROI_input.get_output_at(-1) for ROI_input in ROI_inputs]
    )
    dense_1_with_temperature = keras.layers.Dense(**dense_layer_kwargs)(
        temp_and_image_add
    )
    dense_2_with_temperature = keras.layers.Dense(**dense_layer_kwargs)(
        dense_1_with_temperature
    )
    temperature_aware_do_output = keras.layers.Dense(
        1, activation="relu", kernel_initializer=kernel_initializer
    )(dense_2_with_temperature)

    temperature_aware_model = keras.models.Model(
        inputs=[temperature_input]
        + [ROI_input.get_input_at(0) for ROI_input in ROI_inputs],
        outputs=temperature_aware_do_output,
    )

    temperature_aware_model.compile(
        optimizer=optimizer,
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return temperature_aware_model


def get_hyperparameter_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--convolutional-kernel-size", type=int)
    parser.add_argument("--dense-layer-units", type=int)
    parser.add_argument("--dropblock-size", type=int),
    parser.add_argument("--dropblock-keep-prob", type=int),
    return parser


def main(command_line_args):
    fix_multiprocessing_with_keras_on_macos()

    hyperparameters = get_hyperparameters_from_args(
        command_line_args,
        model_default_hyperparameters=DEFAULT_HYPERPARAMETERS,
        model_hyperparameter_parser=get_hyperparameter_parser(),
    )

    run(hyperparameters, prepare_dataset_ROIs_and_numeric, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

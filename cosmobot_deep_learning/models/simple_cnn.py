"""
This model is a 2-branch network that combines:
1. A hand-made CNN with 3 convolutional layers that trains on full images
2. A dense network that trains on two numeric inputs:
    - temperature
    - numeric output of the hand-made CNN
"""

import argparse
import sys

from tensorflow import keras

from cosmobot_deep_learning.constants import ACTIVATION_LAYER_BY_NAME
from cosmobot_deep_learning.configure import get_model_name_from_filepath
from cosmobot_deep_learning.hyperparameters import (
    get_hyperparameters_from_args,
    get_optimizer,
)
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_image_and_numeric
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)

DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "numeric_input_columns": ["PicoLog temperature (C)"],
    "image_size": 128,
    "convolutional_kernel_size": 5,
    "dense_layer_units": 32,
    "prediction_dense_layer_units": 64,
    "optimizer_name": "adam",
    # 0.0001 learns faster than 0.00001, but 0.0003 and higher causes issues (2019-08-27)
    "learning_rate": 0.0001,
    "dropout_rate": 0.01,
    "output_activation_layer": "sigmoid",
    "convolutional_activation_layer": "leakyrelu",
}


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    # x_train is a list of two inputs: numeric and images
    x_train_numeric, x_train_images = x_train
    x_train_samples_count, numeric_inputs_count = x_train_numeric.shape

    optimizer = get_optimizer(hyperparameters)

    image_size = hyperparameters["image_size"]
    convolutional_kernel_size = hyperparameters["convolutional_kernel_size"]
    convolutional_kernel_shape = (convolutional_kernel_size, convolutional_kernel_size)
    dense_layer_units = hyperparameters["dense_layer_units"]
    prediction_dense_layer_units = hyperparameters["prediction_dense_layer_units"]
    dropout_rate = hyperparameters["dropout_rate"]
    output_activation_layer = ACTIVATION_LAYER_BY_NAME[
        hyperparameters["output_activation_layer"]
    ]
    convolutional_activation_layer = ACTIVATION_LAYER_BY_NAME[
        hyperparameters["convolutional_activation_layer"]
    ]

    kernel_initializer = keras.initializers.he_normal()

    image_to_do_model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                16,
                convolutional_kernel_shape,
                input_shape=(image_size, image_size, 3),
                kernel_initializer=kernel_initializer,
            ),
            convolutional_activation_layer(),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(
                32, convolutional_kernel_shape, kernel_initializer=kernel_initializer
            ),
            convolutional_activation_layer(),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(
                32, convolutional_kernel_shape, kernel_initializer=kernel_initializer
            ),
            convolutional_activation_layer(),
            keras.layers.Flatten(name="prep-for-dense"),
            keras.layers.Dense(
                dense_layer_units,
                activation="relu",
                kernel_initializer=kernel_initializer,
            ),
            keras.layers.Dense(
                dense_layer_units,
                name="final_dense",
                kernel_initializer=kernel_initializer,
            ),
            keras.layers.LeakyReLU(),
            # Final output layer with 1 neuron to regress a single value
            keras.layers.Dense(1, kernel_initializer=kernel_initializer, name="DO"),
        ]
    )

    image_to_do_model.compile(
        optimizer=optimizer,
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    temperature_input = keras.layers.Input(
        shape=(numeric_inputs_count,), name="temperature"
    )

    latest_layer = keras.layers.concatenate(
        [temperature_input, image_to_do_model.get_layer(name="final_dense").output]
    )
    latest_layer = keras.layers.Dense(
        prediction_dense_layer_units, kernel_initializer=kernel_initializer
    )(latest_layer)
    latest_layer = keras.layers.Dropout(dropout_rate)(latest_layer)
    latest_layer = keras.layers.LeakyReLU()(latest_layer)
    latest_layer = keras.layers.Dense(
        prediction_dense_layer_units, kernel_initializer=kernel_initializer
    )(latest_layer)
    latest_layer = keras.layers.LeakyReLU()(latest_layer)
    latest_layer = keras.layers.Dense(
        1, kernel_initializer=kernel_initializer, name="temp-aware-DO"
    )(latest_layer)
    latest_layer = output_activation_layer()(latest_layer)

    temperature_aware_model = keras.models.Model(
        inputs=[temperature_input, image_to_do_model.get_input_at(0)],
        outputs=[latest_layer],
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
    parser.add_argument("--prediction-dense-layer-units", type=int)
    parser.add_argument("--dropout-rate", type=float)
    parser.add_argument(
        "--output-activation-layer", choices=list(ACTIVATION_LAYER_BY_NAME.keys())
    ),
    parser.add_argument(
        "--convolutional-activation-layer",
        choices=list(ACTIVATION_LAYER_BY_NAME.keys()),
    )
    return parser


def main(command_line_args):
    fix_multiprocessing_with_keras_on_macos()

    simple_cnn_hyperparameter_parser = get_hyperparameter_parser()

    hyperparameters = get_hyperparameters_from_args(
        command_line_args, DEFAULT_HYPERPARAMETERS, simple_cnn_hyperparameter_parser
    )

    run(hyperparameters, prepare_dataset_image_and_numeric, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

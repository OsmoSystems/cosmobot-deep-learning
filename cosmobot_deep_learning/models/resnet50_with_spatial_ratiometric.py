"""
This model is a 2-branch network that combines:
1. A pre-trained ResNet50 with new dense layers tacked on that trains on full images
2. A dense network that trains on two numeric inputs:
    - temperature
    - spatial ratiometric ("OO DO patch Wet r_msorm" / "Type 1 Chemistry Hand Applied Dry r_msorm")
"""

import sys

import keras
import keras_resnet.models
import tensorflow as tf

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


PREPARE_DATASET_FUNCTION = prepare_dataset_image_and_numeric
DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "numeric_input_columns": ["sr", "PicoLog temperature (C)"],
    "image_size": 128,
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
    input_layer = keras.layers.Input(shape=(image_size, image_size, 3))

    resnet = keras_resnet.models.ResNet50(input_layer, include_top=False)
    intermediate_resnet_model = keras.Model(
        inputs=resnet.input, outputs=resnet.get_layer("res5c_relu").get_output_at(0)
    )

    residual_model = keras.Sequential(
        [
            intermediate_resnet_model,
            keras.layers.Flatten(),
            keras.layers.Dense(32),
            keras.layers.BatchNormalization(),
            keras.layers.advanced_activations.LeakyReLU(),
            keras.layers.Dense(1, name="residual"),
        ]
    )

    sv_model = keras.models.Sequential(
        [
            keras.layers.Dense(
                11, activation=tf.nn.relu, input_shape=[numeric_inputs_count]
            ),
            keras.layers.Dense(32),
            keras.layers.advanced_activations.LeakyReLU(),
            keras.layers.Dense(1, name="sv_DO"),
        ]
    )

    prediction_with_residual = keras.layers.add(
        [
            sv_model.get_layer(name="sv_DO").output,
            residual_model.get_layer(name="residual").output,
        ],
        name="prediction_with_residual",
    )

    combined_residual_model = keras.models.Model(
        inputs=[sv_model.get_input_at(0), residual_model.get_input_at(0)],
        outputs=prediction_with_residual,
    )

    combined_residual_model.compile(
        optimizer=optimizer,
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return combined_residual_model


def main(command_line_args):
    fix_multiprocessing_with_keras_on_macos()

    hyperparameters = get_hyperparameters_from_args(
        command_line_args, DEFAULT_HYPERPARAMETERS
    )

    run(hyperparameters, PREPARE_DATASET_FUNCTION, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

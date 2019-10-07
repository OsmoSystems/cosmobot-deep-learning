"""
This model is a hand-made CNN with 3 convolutional layers that trains on full images.
"""

import sys

import tensorflow.keras as keras

from cosmobot_deep_learning.configure import get_model_name_from_filepath
from cosmobot_deep_learning.hyperparameters import (
    get_hyperparameters_from_args,
    get_optimizer,
)
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_image_only
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)

DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "numeric_input_columns": [],
    "image_size": 128,
    "optimizer_name": "adam",
    # 0.0001 learns faster than 0.00001, but 0.0003 and higher causes issues (2019-08-27)
    "learning_rate": 0.0001,
}


def create_model(hyperparameters, x_train):
    """ Build a model which will an image data numeric predictions

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (unused)
    """
    image_size = hyperparameters["image_size"]
    optimizer = get_optimizer(hyperparameters)

    kernel_initializer = keras.initializers.he_normal()

    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                16,
                (3, 3),
                activation="relu",
                input_shape=(image_size, image_size, 3),
                kernel_initializer=kernel_initializer,
            ),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", kernel_initializer=kernel_initializer
            ),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", kernel_initializer=kernel_initializer
            ),
            keras.layers.Flatten(name="prep-for-dense"),
            keras.layers.Dense(
                64, activation="relu", kernel_initializer=kernel_initializer
            ),
            keras.layers.Dense(
                64, name="final_dense", kernel_initializer=kernel_initializer
            ),
            keras.layers.LeakyReLU(),
            # Final output layer with 1 neuron to regress a single value
            keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer=kernel_initializer,
                name="DO",
            ),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return model


def main(command_line_args):
    fix_multiprocessing_with_keras_on_macos()

    hyperparameters = get_hyperparameters_from_args(
        command_line_args, DEFAULT_HYPERPARAMETERS
    )

    run(hyperparameters, prepare_dataset_image_only, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

"""
This model is a 2-branch network that combines:
1. A hand-made CNN with 3 convolutional layers that trains on full images
2. A dense network that trains on two numeric inputs:
    - temperature
    - numeric output of the hand-made CNN
"""

import os
import sys

import keras

from cosmobot_deep_learning.configure import (
    parse_model_run_args,
    get_model_name_from_filepath,
)
from cosmobot_deep_learning.hyperparameters import get_hyperparameters
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_image_only
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)

# 0.0001 learns faster than 0.00001, but 0.0003 and higher causes issues (2019-08-27)
LEARNING_RATE = 0.0001


def create_model(hyperparameters, x_train):
    """ Build a model which will an image data numeric predictions

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (unused)
    """
    image_size = hyperparameters["image_size"]

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
            keras.layers.advanced_activations.LeakyReLU(),
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
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return model


if __name__ == "__main__":
    fix_multiprocessing_with_keras_on_macos()

    args = parse_model_run_args(sys.argv[1:])

    # Note: we may eventually need to change how we set this to be compatible with
    # hyperparameter sweeps. See https://www.wandb.com/articles/multi-gpu-sweeps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    hyperparameters = get_hyperparameters(
        model_name=get_model_name_from_filepath(__file__),
        dataset_filename="2019-08-09--14-33-26_osmo_ml_dataset.csv",
        numeric_input_columns=[],
        image_size=128,
        dataset_cache_name=args.dataset_cache,
        optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
        learning_rate=LEARNING_RATE,
    )

    run(
        hyperparameters,
        prepare_dataset_image_only,
        create_model,
        dryrun=args.dryrun,
        dataset_cache_name=args.dataset_cache,
    )

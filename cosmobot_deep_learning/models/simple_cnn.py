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
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_image_and_numeric
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)

# 0.0001 learns faster than 0.00001, but 0.0003 and higher causes issues (2019-08-27)
LEARNING_RATE = 0.0001


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    # x_train is a list of two inputs: numeric and images
    x_train_numeric, x_train_images = x_train
    x_train_samples_count, numeric_inputs_count = x_train_numeric.shape

    image_size = hyperparameters["image_size"]

    kernel_initializer = keras.initializers.he_normal()

    image_to_do_model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                16,
                (3, 3),
                input_shape=(image_size, image_size, 3),
                activation="relu",
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
            keras.layers.Dense(1, kernel_initializer=kernel_initializer, name="DO"),
        ]
    )

    image_to_do_model.compile(
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    temperature_input = keras.layers.Input(
        shape=(numeric_inputs_count,), name="temperature"
    )

    temp_and_image_add = keras.layers.concatenate(
        [temperature_input, image_to_do_model.get_layer(name="final_dense").output]
    )
    dense_1_with_temperature = keras.layers.Dense(
        64, activation="relu", kernel_initializer=kernel_initializer
    )(temp_and_image_add)
    dense_2_with_temperature = keras.layers.Dense(
        64, activation="relu", kernel_initializer=kernel_initializer
    )(dense_1_with_temperature)
    temperature_aware_do_output = keras.layers.Dense(
        1, kernel_initializer=kernel_initializer, name="temp-aware-DO"
    )(dense_2_with_temperature)

    temperature_aware_model = keras.models.Model(
        inputs=[temperature_input, image_to_do_model.get_input_at(0)],
        outputs=[temperature_aware_do_output],
    )

    temperature_aware_model.compile(
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return temperature_aware_model


if __name__ == "__main__":
    fix_multiprocessing_with_keras_on_macos()

    args = parse_model_run_args(sys.argv[1:])

    # Note: we may eventually need to change how we set this to be compatible with
    # hyperparameter sweeps. See https://www.wandb.com/articles/multi-gpu-sweeps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    hyperparameters = get_hyperparameters(
        model_name=get_model_name_from_filepath(__file__),
        numeric_input_columns=["PicoLog temperature (C)"],
        image_size=128,
        dataset_cache_name=args.dataset_cache,
        optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
        learning_rate=LEARNING_RATE,
    )

    run(
        hyperparameters,
        prepare_dataset_image_and_numeric,
        create_model,
        dryrun=args.dryrun,
        dataset_cache_name=args.dataset_cache,
    )

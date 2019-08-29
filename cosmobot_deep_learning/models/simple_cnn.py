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
DEFAULT_LEARNING_RATE = 0.0001

DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "dataset_filename": "2019-08-09--14-33-26_osmo_ml_dataset.csv",
    "numeric_input_columns": ["PicoLog temperature (C)"],
    "image_size": 128,
    # TODO MAJOR: remove this and just take an optimizer_name. generate the actual optimizer when building the model.
    "optimizer": keras.optimizers.Adam(lr=DEFAULT_LEARNING_RATE),
    "learning_rate": DEFAULT_LEARNING_RATE,
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

    image_size = hyperparameters["image_size"]

    kernel_initializer = keras.initializers.he_normal()

    image_to_do_model = keras.models.Sequential(
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
        1,
        activation="sigmoid",
        kernel_initializer=kernel_initializer,
        name="temp-aware-DO",
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


def main(command_line_args):
    fix_multiprocessing_with_keras_on_macos()

    args = parse_model_run_args(command_line_args)

    # for sweeps, this should be set when running the agent(s), example:
    # CUDA_VISIBLE_DEVICES=0 wandb agent mcg70107
    # CUDA_VISIBLE_DEVICES=1 wandb agent mcg70107
    # TODO explode if CUDA_VISIBLE_DEVICES is not defined in environment when --gpu not passed in?
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # TODO hack (we might not want all args in hyperparameters)
    hyperparameters = get_hyperparameters(**{**DEFAULT_HYPERPARAMETERS, **vars(args)})

    # TODO remove
    print(hyperparameters)

    run(
        hyperparameters,
        prepare_dataset_image_and_numeric,
        create_model,
        # TODO we could just read these out of `hyperparameters` instead of passing them (if we're ok with them being in hyperparameters)
        dryrun=args.dryrun,
        dataset_cache_name=args.dataset_cache_name,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

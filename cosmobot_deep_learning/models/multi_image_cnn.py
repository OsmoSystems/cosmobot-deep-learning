"""
This model is an N-branch network that combines:
1. A configurable number of hand-made CNNs with 3 convolutional layers each that train on distinct image inputs
2. A dense network that trains on two numeric inputs:
    - temperature
    - numeric output of the hand-made CNNs
"""

import os
import sys

import keras

from cosmobot_deep_learning.configure import (
    parse_model_run_args,
    get_model_name_from_filepath,
)
from cosmobot_deep_learning.hyperparameters import get_hyperparameters
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_ROIs_and_numeric
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)


def get_convolutional_input(id, x_train_sample_image, kernel_initializer):
    # Layer names cannot have spaces
    model_branch_id = id.replace(" ", "_")

    convolutional_sub_model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                16,
                (3, 3),
                activation="relu",
                input_shape=x_train_sample_image.shape,
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
            keras.layers.Flatten(name=f"{model_branch_id}-prep-for-dense"),
            keras.layers.Dense(
                64, activation="relu", kernel_initializer=kernel_initializer
            ),
            keras.layers.Dense(
                64,
                name=f"{model_branch_id}-final_dense",
                kernel_initializer=kernel_initializer,
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
    image_input_ids = hyperparameters["image_input_ids"]

    # x_train is a list of [numeric_inputs, image_input_1, ..., image_input_x]
    assert len(image_input_ids) == len(x_train) - 1
    x_train_numeric = x_train[0]
    x_train_samples_count, numeric_inputs_count = x_train_numeric.shape

    kernel_initializer = keras.initializers.he_normal()

    image_inputs = [
        get_convolutional_input(image_input_id, x_train[i + 1][0], kernel_initializer)
        for i, image_input_id in enumerate(image_input_ids)
    ]

    temperature_input = keras.layers.Input(
        shape=(numeric_inputs_count,), name="temperature"
    )

    temp_and_image_add = keras.layers.concatenate(
        [temperature_input]
        + [image_input.get_output_at(-1) for image_input in image_inputs]
    )
    dense_1_with_temperature = keras.layers.Dense(
        64, activation="relu", kernel_initializer=kernel_initializer
    )(temp_and_image_add)
    dense_2_with_temperature = keras.layers.Dense(
        64, activation="relu", kernel_initializer=kernel_initializer
    )(dense_1_with_temperature)
    temperature_aware_do_output = keras.layers.Dense(
        1, activation="sigmoid", kernel_initializer=kernel_initializer
    )(dense_2_with_temperature)

    temperature_aware_model = keras.models.Model(
        inputs=[temperature_input]
        + [image_input.get_input_at(0) for image_input in image_inputs],
        outputs=temperature_aware_do_output,
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
        dataset_filename="2019-08-27--12-24-59_osmo_ml_dataset.csv",
        numeric_input_columns=["PicoLog temperature (C)"],
        image_size=64,
        dataset_cache_name=args.dataset_cache,
        # ROI names to extract from `ROI definitions` column in the dataset.
        # WARNING: The order here is preserved through data processing and model creation / input
        # If you are using a cached dataset, make sure you have the correct order.
        image_input_ids=["DO patch", "reference patch", "reflectance standard"],
    )

    run(
        hyperparameters,
        prepare_dataset_ROIs_and_numeric,
        create_model,
        dryrun=args.dryrun,
        dataset_cache_name=args.dataset_cache,
    )

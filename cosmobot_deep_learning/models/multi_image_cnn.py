"""
This model is a 4-branch network that combines:
1. 3 hand-made CNNs with 3 convolutional layers each that train on ROI crops
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
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_rois_and_numeric
from cosmobot_deep_learning.run import run


def get_convolutional_input(name, hyperparameters, kernel_initializer):
    image_size = hyperparameters["image_size"]

    convolutional_sub_model = keras.models.Sequential(
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
            keras.layers.Flatten(name=f"{name}-prep-for-dense"),
            keras.layers.Dense(
                64, activation="relu", kernel_initializer=kernel_initializer
            ),
            keras.layers.Dense(
                64, name=f"{name}-final_dense", kernel_initializer=kernel_initializer
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
    # x_train is a list of four inputs: numeric and image ROIs
    x_train_numeric, x_train_roi_1, x_train_roi_2, x_train_roi_3 = x_train
    x_train_samples_count, numeric_inputs_count = x_train_numeric.shape

    kernel_initializer = keras.initializers.he_normal()

    image_input_1 = get_convolutional_input(
        "roi-1", hyperparameters, kernel_initializer
    )

    image_input_2 = get_convolutional_input(
        "roi-2", hyperparameters, kernel_initializer
    )

    image_input_3 = get_convolutional_input(
        "roi-3", hyperparameters, kernel_initializer
    )

    temperature_input = keras.layers.Input(
        shape=(numeric_inputs_count,), name="temperature"
    )

    temp_and_image_add = keras.layers.concatenate(
        [
            temperature_input,
            image_input_1.get_output_at(-1),
            image_input_2.get_output_at(-1),
            image_input_3.get_output_at(-1),
        ]
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
        inputs=[
            temperature_input,
            image_input_1.get_input_at(0),
            image_input_2.get_input_at(0),
            image_input_3.get_input_at(0),
        ],
        outputs=temperature_aware_do_output,
    )

    temperature_aware_model.compile(
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return temperature_aware_model


if __name__ == "__main__":
    args = parse_model_run_args(sys.argv[1:])

    # Note: we may eventually need to change how we set this to be compatible with
    # hyperparameter sweeps. See https://www.wandb.com/articles/multi-gpu-sweeps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    hyperparameters = get_hyperparameters(
        model_name=get_model_name_from_filepath(__file__),
        dataset_filename="2019-08-23--15-00-40_osmo_ml_dataset.csv",
        numeric_input_columns=["PicoLog temperature (C)"],
        image_size=64,
    )

    run(
        hyperparameters,
        prepare_dataset_rois_and_numeric,
        create_model,
        dryrun=args.dryrun,
    )

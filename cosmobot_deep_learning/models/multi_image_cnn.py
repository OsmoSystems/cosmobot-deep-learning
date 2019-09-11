"""
This model is an N-branch network that combines:
1. A configurable number of hand-made CNNs with 3 convolutional layers each that train on distinct ROI inputs
2. A dense network that trains on numeric inputs:
    - temperature
    - numeric output of each of the hand-made CNNs
"""

import sys

import keras

from cosmobot_deep_learning.configure import get_model_name_from_filepath
from cosmobot_deep_learning.gpu_configuration import dont_use_all_the_gpu_memory
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
    "image_size": 64,
    # ROI names to extract from `ROI definitions` column in the dataset.
    # WARNING: The order here is preserved through data processing and model creation / input
    # If you are using a cached dataset, make sure you have the correct order.
    "input_ROI_names": ["DO patch", "reference patch", "reflectance standard"],
}


def get_convolutional_input(branch_id, x_train_sample_image, kernel_initializer):
    # Layer names cannot have spaces
    model_branch_id = branch_id.replace(" ", "_")

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
    input_ROI_names = hyperparameters["input_ROI_names"]
    optimizer = get_optimizer(hyperparameters)

    # x_train is a list of [numeric_inputs, ROI_input_1, ..., ROI_input_x]
    assert len(input_ROI_names) == len(x_train) - 1
    x_train_numeric = x_train[0]
    x_train_samples_count, numeric_inputs_count = x_train_numeric.shape

    kernel_initializer = keras.initializers.he_normal()

    ROI_inputs = [
        get_convolutional_input(ROI_name, x_train[i + 1][0], kernel_initializer)
        for i, ROI_name in enumerate(input_ROI_names)
    ]

    temperature_input = keras.layers.Input(
        shape=(numeric_inputs_count,), name="temperature"
    )

    temp_and_image_add = keras.layers.concatenate(
        [temperature_input] + [ROI_input.get_output_at(-1) for ROI_input in ROI_inputs]
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
        + [ROI_input.get_input_at(0) for ROI_input in ROI_inputs],
        outputs=temperature_aware_do_output,
    )

    temperature_aware_model.compile(
        optimizer=optimizer,
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return temperature_aware_model


def main(command_line_args):
    dont_use_all_the_gpu_memory()
    fix_multiprocessing_with_keras_on_macos()

    hyperparameters = get_hyperparameters_from_args(
        command_line_args, DEFAULT_HYPERPARAMETERS
    )

    run(hyperparameters, prepare_dataset_ROIs_and_numeric, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

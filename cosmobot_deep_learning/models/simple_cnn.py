"""
This model is a 2-branch network that combines:
1. A hand-made CNN with 3 convolutional layers that trains on full images
2. A dense network that trains on two numerical inputs:
    - temperature
    - numerical output of the hand-made CNN
"""

import os
import sys

import keras
import pandas as pd
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import (
    load_multi_experiment_dataset_csv,
    get_pkg_dataset_filepath,
    get_dataset_hash,
)
from cosmobot_deep_learning.configure import (
    parse_model_run_args,
    get_model_name_from_filepath,
)
from cosmobot_deep_learning.constants import (
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG,
    ACCEPTABLE_ERROR_MG_L,
    ACCEPTABLE_ERROR_MMHG,
)
from cosmobot_deep_learning.custom_metrics import (
    get_fraction_outside_acceptable_error_fn,
    magical_incantation_to_make_custom_metric_work,
)
from cosmobot_deep_learning.preprocess_image import open_and_preprocess_images


_DATASET_FILENAME = "2019-08-09--14-33-26_osmo_ml_dataset.csv"
_DATASET_FILEPATH = get_pkg_dataset_filepath(_DATASET_FILENAME)


# Normalize by the atmospheric partial pressure of oxygen, as that is roughly the max we expect
LABEL_SCALE_FACTOR_MMHG = ATMOSPHERIC_OXYGEN_PRESSURE_MMHG
LABEL_COLUMN_NAME = "YSI DO (mmHg)"

# Ensure that our custom metric uses the same normalizing factor we use to scale our labels
_ACCEPTABLE_ERROR_NORMALIZED = ACCEPTABLE_ERROR_MMHG / LABEL_SCALE_FACTOR_MMHG
fraction_outside_acceptable_error = get_fraction_outside_acceptable_error_fn(
    acceptable_error=_ACCEPTABLE_ERROR_NORMALIZED
)

_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "dataset_filename": _DATASET_FILENAME,
    "dataset_filepath": _DATASET_FILEPATH,
    "dataset_hash": get_dataset_hash(_DATASET_FILEPATH),
    "epochs": 1000,
    "batch_size": 128,
    "image_size": 128,
    "optimizer": keras.optimizers.Adadelta(),
    "loss": "mean_squared_error",
    # Toss in all the constants / assumptions we're using in this run
    "ACCEPTABLE_ERROR_MG_L": ACCEPTABLE_ERROR_MG_L,
    "ACCEPTABLE_ERROR_MMHG": ACCEPTABLE_ERROR_MMHG,
    "LABEL_SCALE_FACTOR_MMHG": LABEL_SCALE_FACTOR_MMHG,
    "LABEL_COLUMN_NAME": LABEL_COLUMN_NAME,
    "_ACCEPTABLE_ERROR_NORMALIZED": _ACCEPTABLE_ERROR_NORMALIZED,
}


def extract_input_params(df):
    """ Get the non-image input data values

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
        Returns:
            Numpy array of temperature values.
    """
    normalized_dataset = pd.DataFrame(
        {"PicoLog temperature (C)": df["PicoLog temperature (C)"]}
    )

    return normalized_dataset.values


def extract_label_values(df):
    """ Get the label (y) data values for a given dataset (x)

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
        Returns:
            Numpy array of dissolved oxygen label values, normalized by a constant scale factor
    """
    scaled_labels = df[LABEL_COLUMN_NAME] / LABEL_SCALE_FACTOR_MMHG
    return scaled_labels.values


def prepare_dataset(raw_dataset, input_image_dimension):
    """ Transform a dataset CSV into the appropriate inputs for training the model in this module.

        Args:
            raw_dataset: A DataFrame corresponding to a standard cosmobot dataset CSV
            input_image_dimension: The desired side length of the output (square) images
        Returns:
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """
    train_samples = raw_dataset[raw_dataset["training_resampled"]]
    test_samples = raw_dataset[raw_dataset["test"]]

    x_train_sr = extract_input_params(train_samples)
    x_train_images = open_and_preprocess_images(
        train_samples["local_filepath"].values, input_image_dimension
    )
    y_train_do = extract_label_values(train_samples)

    x_test_sr = extract_input_params(test_samples)
    x_test_images = open_and_preprocess_images(
        test_samples["local_filepath"].values, input_image_dimension
    )
    y_test_do = extract_label_values(test_samples)

    return (
        [x_train_sr, x_train_images],
        [y_train_do],
        [x_test_sr, x_test_images],
        [y_test_do],
    )


def create_model(hyperparameters, input_numerical_data_dimension):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        input_numerical_data_dimension: The number of numerical inputs to feed to the model
    """
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
        metrics=[
            "mean_squared_error",
            "mean_absolute_error",
            fraction_outside_acceptable_error,
        ],
    )

    temperature_input = keras.layers.Input(
        shape=(input_numerical_data_dimension,), name="temperature"
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
        metrics=[
            "mean_squared_error",
            "mean_absolute_error",
            fraction_outside_acceptable_error,
        ],
    )

    return temperature_aware_model


def run(
    epochs: int,
    batch_size: int,
    model_name: str,
    dataset_filepath: str,
    **additional_hyperparameters
):
    """ Use the provided hyperparameters to train the model in this module.

        Args:
            epochs: Number of epochs to train for
            batch_size: Training input batch size
            model_name: A string label for the model. Should match this module name
            dataset_filepath: Filepath (within this package) of the dataset resource to load and train with
            additional_hyperparameters: Any other variables that are parameterizable for this model
                image_size: The desired side length for resized (square) images
                optimizer: Which optimizer function to use
                loss: Which loss function to use

    """

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=load_multi_experiment_dataset_csv(dataset_filepath),
        input_image_dimension=additional_hyperparameters["image_size"],
    )

    # Report dataset sizes to wandb now that we know them
    wandb.config.train_sample_count = y_train[0].shape[0]
    wandb.config.test_sample_count = y_test[0].shape[0]

    x_train_sr = x_train[0]

    model = create_model(
        hyperparameters=additional_hyperparameters,
        input_numerical_data_dimension=x_train_sr.shape[1],
    )

    magical_incantation_to_make_custom_metric_work()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()],
    )

    return history


if __name__ == "__main__":
    args = parse_model_run_args(sys.argv[1:])

    # Note: we may eventually need to change how we set this to be compatible with
    # hyperparameter sweeps. See https://www.wandb.com/articles/multi-gpu-sweeps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    wandb.init(
        # Add a "jupyter" tag when running from jupyter notebooks
        tags=[],
        config=_HYPERPARAMETERS,
    )

    run(**_HYPERPARAMETERS)

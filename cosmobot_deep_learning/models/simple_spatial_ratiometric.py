"""
This model is a dense-layer network that trains only on two numerical inputs:
 - temperature
 - spatial ratiometric ("OO DO patch Wet r_msorm" / "Type 1 Chemistry Hand Applied Dry r_msorm")
"""

import os
import sys

import keras
import pandas as pd
import tensorflow as tf
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

_DATASET_FILENAME = "2019-08-08--11-09-20_osmo_ml_dataset.csv"
_DATASET_FILEPATH = get_pkg_dataset_filepath(_DATASET_FILENAME)


# Normalize by the atmospheric partial pressure of oxygen, as that is roughly the max we expect
LABEL_SCALE_FACTOR_MMHG = ATMOSPHERIC_OXYGEN_PRESSURE_MMHG

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
    "epochs": 10000,
    "batch_size": 125,
    "optimizer": keras.optimizers.Adadelta(),
    "loss": "mean_squared_error",
    # Toss in all the constants / assumptions we're using in this run
    "ACCEPTABLE_ERROR_MG_L": ACCEPTABLE_ERROR_MG_L,
    "ACCEPTABLE_ERROR_MMHG": ACCEPTABLE_ERROR_MMHG,
    "LABEL_SCALE_FACTOR_MMHG": LABEL_SCALE_FACTOR_MMHG,
    "_ACCEPTABLE_ERROR_NORMALIZED": _ACCEPTABLE_ERROR_NORMALIZED,
}


def extract_input_params(df):
    """ Get the non-image input data values

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
        Returns:
            Numpy array of temperature and spatial ratiometric values.
            Ratiometric value is Wet DO Patch / Dry Reference Patch.
    """
    normalized_dataset = pd.DataFrame(
        {
            # Keep math on the same line
            # fmt: off
            "PicoLog temperature (C)": df["PicoLog temperature (C)"],
            "spatial_ratiometric": df["DO patch r_msorm"] / df["reference patch r_msorm"],
            # fmt: on
        }
    )

    return normalized_dataset.values


def extract_label_values(df):
    """ Get the label (y) data values for a given dataset (x)

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
        Returns:
            Numpy array of dissolved oxygen label values, normalized by a constant scale factor
    """
    scaled_labels = df["YSI DO (mmHg)"] / LABEL_SCALE_FACTOR_MMHG
    return scaled_labels.values


def prepare_dataset(raw_dataset):
    """ Transform a dataset CSV into the appropriate inputs for training the model in this module.

        Args:
            raw_dataset: A DataFrame corresponding to a standard cosmobot dataset csv
        Returns:
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """

    train_samples = raw_dataset[raw_dataset["training_resampled"]]
    test_samples = raw_dataset[raw_dataset["test"]]

    x_train_sr = extract_input_params(train_samples)
    y_train_do = extract_label_values(train_samples)

    x_test_sr = extract_input_params(test_samples)
    y_test_do = extract_label_values(test_samples)

    return (x_train_sr, y_train_do, x_test_sr, y_test_do)


def create_model(hyperparameters, input_numerical_data_dimensions):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        input_numerical_data_dimension: The number of numerical inputs to feed to the model
    """
    sr_model = keras.models.Sequential(
        [
            keras.layers.Dense(
                11, activation=tf.nn.relu, input_shape=[input_numerical_data_dimensions]
            ),
            keras.layers.Dense(32),
            keras.layers.advanced_activations.LeakyReLU(),
            keras.layers.Dense(1, name="sv_DO"),
        ]
    )

    sr_model.compile(
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=[
            "mean_squared_error",
            "mean_absolute_error",
            fraction_outside_acceptable_error,
        ],
    )

    return sr_model


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
                optimizer: Which optimizer function to use
                loss: Which loss function to use

    """
    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=load_multi_experiment_dataset_csv(dataset_filepath)
    )

    # Report dataset sizes to wandb now that we know them
    wandb.config.train_sample_count = y_train.shape[0]
    wandb.config.test_sample_count = y_test.shape[0]

    model = create_model(
        additional_hyperparameters, input_numerical_data_dimensions=x_train.shape[1]
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

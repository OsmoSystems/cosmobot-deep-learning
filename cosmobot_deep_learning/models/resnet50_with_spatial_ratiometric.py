from functools import partial
import os
import sys

import keras
import keras_resnet.models
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
)
from cosmobot_deep_learning.preprocess_image import open_and_preprocess_images


_DATASET_FILENAME = "2019-06-27--08-24-58_osmo_ml_dataset.csv"
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
    "image_size": 128,
    "optimizer": keras.optimizers.Adadelta(),
    "loss": "mean_squared_error",
    "metrics": [
        "mean_squared_error",
        "mean_absolute_error",
        fraction_outside_acceptable_error,
    ],
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
            "temperature_set_point": df["temperature_set_point"],
            "spatial_ratiometric": df["OO DO patch Wet r_msorm"] / df["Type 1 Chemistry Hand Applied Dry r_msorm"],
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
    scaled_labels = df["YSI Dissolved Oxygen (mmHg)"] / LABEL_SCALE_FACTOR_MMHG
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
    # TODO: consider changing from "test" to "dev" (or "val"?).
    # Would require updating dataset csv columne headers too
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


def create_model(hyperparameters, input_numerical_data_dimensions):
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
                11, activation=tf.nn.relu, input_shape=[input_numerical_data_dimensions]
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
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return combined_residual_model


# TODO: I made some of these explicit parameters - the ones I thought would definitely be shared between all models
# But maybe it would be better to just pass a single `hyperparameters` dict?
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
    wandb.config.train_sample_count = y_train.shape[0]
    wandb.config.test_sample_count = y_test.shape[0]

    x_train_sr_shape = x_train[0].shape

    model = create_model(
        additional_hyperparameters, input_numerical_data_dimensions=x_train_sr_shape[1]
    )

    # TODO: spend 5 minutes trying to find out why this works
    # I honestly have no idea why this makes our custom metric work... but it does.
    # https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    keras.backend.get_session().run(tf.local_variables_initializer())

    history = model.fit(
        x_train,
        y_train,
        epochs,
        batch_size,
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

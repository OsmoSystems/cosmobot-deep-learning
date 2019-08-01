import os
import pkg_resources
import sys

import keras
import keras_resnet.models
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import load_multi_experiment_dataset_csv
from cosmobot_deep_learning.configure import parse_model_run_args
from cosmobot_deep_learning.preprocess_image import open_crop_and_scale_image

PACKAGE_NAME = "cosmobot_deep_learning"

_DEFAULT_INPUT_IMAGE_DIMENSIONS = 128
DO_SCALE_FACTOR = 160
BAROMETER_SCALE_FACTOR = 800


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
            "temperature_set_point":
                df["temperature_set_point"],
            "spatial_ratiometric":
                df["OO DO patch Wet r_msorm"] / df["Type 1 Chemistry Hand Applied Dry r_msorm"],
            # fmt: on
        }
    )

    return normalized_dataset.values


def prepare_input_images(image_filepaths, image_dimenstions):
    """ Preprocess the input images and prepare them for direct use in training a model

        Args:
            image_filepaths: An iterable list of filepaths to images to prepare
        Returns:
            A single numpy array of all images resized to the appropriate dimensions and concatenated
    """
    return np.array(
        [
            open_crop_and_scale_image(image_filepath, output_size=image_dimenstions)
            for image_filepath in tqdm(image_filepaths)
        ]
    )


def extract_label_values(df):
    """ Get the label (y) data values for a given dataset (x)

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
        Returns:
            Numpy array of disolved oxygen label values, normalized by a constant scale factor
    """
    scaled_labels = df["YSI Dissolved Oxygen (mmHg)"] / DO_SCALE_FACTOR
    return scaled_labels.values


def prepare_dataset(image_data, input_image_dimensions):
    """ Transform a dataset CSV into the appropriate inputs for training the model in this module.

        Args:
            image_data: A DataFrame corresponding to a standard cosmobot dataset CSV
        Returns:
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """
    train_samples = image_data[image_data["training_resampled"]]
    test_samples = image_data[image_data["test"]]

    x_train_sr = extract_input_params(train_samples)
    x_train_images = prepare_input_images(
        train_samples["local_filepath"].values, input_image_dimensions
    )
    y_train_do = extract_label_values(train_samples)

    x_test_sr = extract_input_params(test_samples)
    x_test_images = prepare_input_images(
        test_samples["local_filepath"].values, input_image_dimensions
    )
    y_test_do = extract_label_values(test_samples)

    return (
        [x_train_sr, x_train_images],
        [y_train_do],
        [x_test_sr, x_test_images],
        [y_test_do],
    )


def create_model(input_numerical_data_dimensions, input_image_dimensions):
    input_layer = keras.layers.Input(
        shape=(input_image_dimensions, input_image_dimensions, 3)
    )

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
        optimizer=keras.optimizers.Adadelta(),
        loss="mean_squared_error",
        metrics=["mean_squared_error", "mean_absolute_error"],
    )

    return combined_residual_model


def run(
    epochs,
    batch_size,
    dataset_name,
    input_image_dimensions=_DEFAULT_INPUT_IMAGE_DIMENSIONS,
):
    """ Load the given dataset and use it to train the model in this module.

        Args:
            epochs: Number of epochs to train for
            batch_size: Training input batch size
            dataset_name: Filename of the dataset resource to load and train with
    """
    resource_path = "/".join(["datasets", dataset_name])
    dataset_filepath = pkg_resources.resource_filename(PACKAGE_NAME, resource_path)

    raw_dataset = load_multi_experiment_dataset_csv(dataset_filepath)

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset, input_image_dimensions
    )

    x_train_sr_shape = x_train[0].shape

    model = create_model(
        input_numerical_data_dimensions=x_train_sr_shape[1],
        input_image_dimensions=input_image_dimensions,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()],
    )

    return history


if __name__ == "__main__":

    args = parse_model_run_args(sys.argv[1:])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    wandb.init()
    # Add cli args to w&b config
    wandb.config.update(args)

    run(
        wandb.config.epochs,
        wandb.config.batch_size,
        "2019-06-27--08-24-58_osmo_ml_dataset.csv",
    )
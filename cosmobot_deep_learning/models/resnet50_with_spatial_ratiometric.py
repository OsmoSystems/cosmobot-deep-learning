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
)
from cosmobot_deep_learning.configure import (
    parse_model_run_args,
    get_model_name_from_filepath,
)
from cosmobot_deep_learning.preprocess_image import open_and_preprocess_images


_DEFAULT_INPUT_IMAGE_SIZE = 128
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


def extract_label_values(df):
    """ Get the label (y) data values for a given dataset (x)

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
        Returns:
            Numpy array of disolved oxygen label values, normalized by a constant scale factor
    """
    scaled_labels = df["YSI Dissolved Oxygen (mmHg)"] / DO_SCALE_FACTOR
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

    # TODO: figure out how to create custom metrics & add ours:
    # - % of predictions outside 0.5 mg/l error
    # - Mean Absolute Error

    combined_residual_model.compile(
        optimizer=hyperparameters["optimizer"],
        loss=hyperparameters["loss"],
        # TODO: should this be defined in hyperparameters too?
        # Since it's just what get reported, it seems like it would be kind of moot to report
        # that we're going to report them
        metrics=["mean_squared_error", "mean_absolute_error"],
    )

    return combined_residual_model


# TODO: consider switching to having explicit parameters here so that they can have type checking, etc.
# In that case, I'd probably still pass some params along in a bag of params.
def run(hyperparameters):
    """ Use the provided hyperparameters to train the model in this module.

        Args:
            hyperparameters: A dictionary of hyperparameter config, including:
                dataset_name: Filename of the dataset resource to load and train with
                epochs: Number of epochs to train for
                batch_size: Training input batch size
                image_size: The desired side length for resized (square) images
                optimizer: Which optimizer function to use
                loss: Which loss function to use

    """
    dataset_filepath = get_pkg_dataset_filepath(hyperparameters["dataset_name"])

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=load_multi_experiment_dataset_csv(dataset_filepath),
        input_image_dimension=hyperparameters["image_size"],
    )

    x_train_sr_shape = x_train[0].shape

    model = create_model(
        hyperparameters, input_numerical_data_dimensions=x_train_sr_shape[1]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=hyperparameters["epochs"],
        batch_size=hyperparameters["batch_size"],
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()],
    )

    return history


if __name__ == "__main__":

    args = parse_model_run_args(sys.argv[1:])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    wandb.init()
    wandb.config.update(
        {
            "model_name": get_model_name_from_filepath(__file__),
            "dataset_name": "2019-06-27--08-24-58_osmo_ml_dataset.csv",
            "image_size": _DEFAULT_INPUT_IMAGE_SIZE,
            "optimizer": keras.optimizers.Adadelta(),
            "loss": "mean_squared_error",
        }
    )
    wandb.config.update(args)

    run(hyperparameters=wandb.config)

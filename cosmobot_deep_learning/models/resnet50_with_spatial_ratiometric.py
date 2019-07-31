import os
import sys

import keras
import keras_resnet.models
import tensorflow as tf

from tqdm import tqdm
import numpy as np
import pandas as pd

import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import load_multi_experiment_dataset_csv
from cosmobot_deep_learning.configure import parse_args
from cosmobot_deep_learning.preprocess_image import open_crop_and_scale_image

_TARGET_IMAGE_DIMENSIONS = 128
DO_SCALE_FACTOR = 160
BAROMETER_SCALE_FACTOR = 800


def extract_input_params(df):
    """ Get the non-image input data values

        Args:
            df: A DataFrame representing a dataset
        Returns:
            Numpy array of input values for the given dataset
    """
    normalized_dataset = pd.DataFrame(
        {"temperature_set_point": df["temperature_set_point"]}
    )
    normalized_dataset["norm_baro"] = (
        df["YSI Barometer (mmHg)"] / BAROMETER_SCALE_FACTOR
    )
    normalized_dataset["sr"] = (
        df["OO DO patch Wet r_msorm"] / df["Type 1 Chemistry Hand Applied Dry r_msorm"]
    )

    return normalized_dataset.values


def extract_target_params(df):
    """ Get the target or label data values

        Args:
            df: A DataFrame representing a dataset
        Returns:
            Numpy array of target values for the given dataset
    """
    scaled_targets = df["YSI Dissolved Oxygen (mmHg)"] / DO_SCALE_FACTOR
    return scaled_targets.values


def prepare_dataset(image_data):
    """ Transform a dataset CSV into the appropriate inputs for training the model in this module.

        Args:
            image_data: A DataFrame corresponding to a raw dataset CSV
        Returns:
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """
    train_samples = image_data[image_data["training_resampled"]]
    test_samples = image_data[image_data["test"]]

    x_train_sr = extract_input_params(train_samples)
    x_train_images = np.array(
        [
            open_crop_and_scale_image(
                image_filepath, output_size=_TARGET_IMAGE_DIMENSIONS
            )
            for image_filepath in tqdm(train_samples["local_filepath"].values)
        ]
    )
    y_train_do = extract_target_params(train_samples)

    x_test_sr = extract_input_params(test_samples)
    x_test_images = np.array(
        [
            open_crop_and_scale_image(
                image_filepath, output_size=_TARGET_IMAGE_DIMENSIONS
            )
            for image_filepath in tqdm(test_samples["local_filepath"].values)
        ]
    )
    y_test_do = extract_target_params(test_samples)

    return (
        [x_train_sr, x_train_images],
        [y_train_do],
        [x_test_sr, x_test_images],
        [y_test_do],
    )


def create_model():
    input_layer = keras.layers.Input(
        shape=(_TARGET_IMAGE_DIMENSIONS, _TARGET_IMAGE_DIMENSIONS, 3)
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
            keras.layers.Dense(11, activation=tf.nn.relu, input_shape=[3]),
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
        metrics=["mean_absolute_error"],
    )

    return combined_residual_model


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    if args["gpu"] != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])

    model_directory = os.path.dirname(__file__)
    dataset_filepath = os.path.join(
        model_directory, "../datasets/2019-06-27--08-24-58_osmo_ml_dataset.csv"
    )

    image_data = load_multi_experiment_dataset_csv(dataset_filepath)

    x_train, y_train, x_test, y_test = prepare_dataset(image_data)

    wandb.init()
    # Add cli args to w&b config
    wandb.config.update(args)

    model = create_model()

    history = model.fit(
        x_train,
        y_train,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()],
    )

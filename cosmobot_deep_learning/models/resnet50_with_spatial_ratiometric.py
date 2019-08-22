"""
This model is a 2-branch network that combines:
1. A pre-trained ResNet50 with new dense layers tacked on that trains on full images
2. A dense network that trains on two numerical inputs:
    - temperature
    - spatial ratiometric ("OO DO patch Wet r_msorm" / "Type 1 Chemistry Hand Applied Dry r_msorm")
"""

# TODO: this is working yet!

import os
import sys

import keras
import keras_resnet.models
import tensorflow as tf

from cosmobot_deep_learning.configure import (
    parse_model_run_args,
    get_model_name_from_filepath,
)
from cosmobot_deep_learning.hyperparameters import get_hyperparameters
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_image_and_numerical
from cosmobot_deep_learning.run import run


def create_model(hyperparameters, input_numerical_data_dimension):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        input_numerical_data_dimension: The number of numerical inputs to feed to the model
    """
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
                11, activation=tf.nn.relu, input_shape=[input_numerical_data_dimension]
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


if __name__ == "__main__":
    args = parse_model_run_args(sys.argv[1:])

    # Note: we may eventually need to change how we set this to be compatible with
    # hyperparameter sweeps. See https://www.wandb.com/articles/multi-gpu-sweeps
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # TODO: remove
    os.environ["WANDB_MODE"] = "dryrun"

    hyperparameters = get_hyperparameters(
        model_name=get_model_name_from_filepath(__file__),
        # TODO: revert these for-testing changes
        dataset_filename="2019-08-09--14-33-26_osmo_ml_dataset_tiny.csv",
        epochs=1,
        # End TODO
        input_columns=["sr", "PicoLog temperature (C)"],
        image_size=128,
    )

    run(hyperparameters, prepare_dataset_image_and_numerical, create_model)

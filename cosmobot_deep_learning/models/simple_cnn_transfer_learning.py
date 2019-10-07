"""
This model adds additional trainable dense layers on top of a pre-trained model
"""

import argparse
import sys

from tf import keras
from tf.keras.layers import Dense

from cosmobot_deep_learning.constants import ACTIVATION_LAYER_BY_NAME
from cosmobot_deep_learning.configure import get_model_name_from_filepath
from cosmobot_deep_learning.hyperparameters import (
    get_hyperparameters_from_args,
    get_optimizer,
)
from cosmobot_deep_learning.prepare_dataset import prepare_dataset_image_and_numeric
from cosmobot_deep_learning.run import run
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)

DEFAULT_HYPERPARAMETERS = {
    "model_name": get_model_name_from_filepath(__file__),
    "dataset_filename": "2019-10-03--10-40-27_unit_A_scum_and_calibration.csv",
    "numeric_input_columns": ["YSI temperature (C)"],
    "image_size": 128,
    "convolutional_kernel_size": 5,
    "dense_layer_units": 32,
    "prediction_dense_layer_units": 64,
    "optimizer_name": "adam",
    # 0.0001 learns faster than 0.00001, but 0.0003 and higher causes issues (2019-08-27)
    "learning_rate": 0.0001,
    "dropout_rate": 0.01,
    "output_activation_layer": "sigmoid",
    "convolutional_activation_layer": "leakyrelu",
}


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    # HACKS HACKS HACKS HACKS
    # Have to re-define custom metrics
    from cosmobot_deep_learning.custom_metrics import (
        get_fraction_outside_error_threshold_fn,
    )

    error_thresholds_mg_l = hyperparameters["error_thresholds_mg_l"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]

    fraction_outside_error_threshold_fns = [
        get_fraction_outside_error_threshold_fn(
            error_threshold_mg_l, label_scale_factor_mmhg
        )
        for error_threshold_mg_l in error_thresholds_mg_l
    ]

    custom_objects = {
        **{fn.__name__: fn for fn in fraction_outside_error_threshold_fns}
    }

    # END HACKS

    original_model = keras.models.load_model("0oh4ovjz-model-best.h5", custom_objects)

    original_last_layer_index = 18

    # Start with the output of the last dense layer actication
    last_desired_layer_from_original_model = original_model.layers[
        original_last_layer_index
    ].output
    print(last_desired_layer_from_original_model)

    # Define some new dense layers
    x = last_desired_layer_from_original_model
    x = Dense(512, activation="relu", name="xfer_dense_1")(x)
    x = Dense(256, activation="relu", name="xfer_dense_2")(x)
    x = Dense(128, activation="relu", name="xfer_dense_3")(x)
    output_layer = Dense(1, activation="sigmoid", name="xfer_output")(x)

    transfer_learning_model = keras.models.Model(
        inputs=original_model.input, outputs=output_layer
    )

    # Freeze the original model's layers
    for layer in transfer_learning_model.layers[:original_last_layer_index]:
        layer.trainable = False
    for layer in transfer_learning_model.layers[original_last_layer_index:]:
        layer.trainable = True

    print(transfer_learning_model.summary())

    transfer_learning_model.compile(
        optimizer=get_optimizer(hyperparameters),
        loss=hyperparameters["loss"],
        metrics=hyperparameters["metrics"],
    )

    return transfer_learning_model


def get_hyperparameter_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image-size", type=int)
    return parser


def main(command_line_args):
    fix_multiprocessing_with_keras_on_macos()

    simple_cnn_hyperparameter_parser = get_hyperparameter_parser()

    hyperparameters = get_hyperparameters_from_args(
        command_line_args, DEFAULT_HYPERPARAMETERS, simple_cnn_hyperparameter_parser
    )

    run(hyperparameters, prepare_dataset_image_and_numeric, create_model)


if __name__ == "__main__":
    main(sys.argv[1:])

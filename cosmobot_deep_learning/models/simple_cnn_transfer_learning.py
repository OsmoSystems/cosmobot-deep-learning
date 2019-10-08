"""
This model adds additional trainable dense layers on top of a pre-trained model
"""

import argparse
import sys

from tensorflow import keras

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
    # "dataset_filename": "2019-10-03--10-40-27_unit_A_scum_and_calibration.csv",
    "dataset_filename": "2019-09-26--14-19-22_osmo_ml_dataset_scum_dev_at_start.csv",
    "training_set_column": "cosmobot_a_training_resampled",
    "dev_set_column": "cosmobot_a_dev",
    "numeric_input_columns": ["YSI temperature (C)"],
    "image_size": 128,
    # 0.0001 learns faster than 0.00001, but 0.0003 and higher causes issues (2019-08-27)
    # "learning_rate": 0.0001,
    # Use an even smaller learning rate, so that we don't unlearn old things before we've learned new things
    "learning_rate": 0.00001,
}


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    # HACKS HACKS HACKS HACKS
    # Have to re-define custom metrics so we can re-hydrate the model
    from cosmobot_deep_learning.custom_metrics import (
        get_fraction_outside_error_threshold_fn,
    )

    error_thresholds_mg_l = [0.1, 0.3, 0.5]
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

    # Load out best simple_cnn model that has been trained on calibration data
    # TODO: use wandb API to download this?
    model_filename = f'{hyperparameters["original_model_id"]}-model-best.h5'
    transfer_learning_model = keras.models.load_model(model_filename, custom_objects)

    # The 18th index should be the last dense layer before output: "leaky_re_lu_6"
    # original_last_layer_index = 18
    #
    # # Start with the output of the last dense layer activation
    # last_desired_layer_from_original_model = original_model.layers[
    #     original_last_layer_index
    # ].output
    # print(last_desired_layer_from_original_model)
    #
    # # Define some new dense layers
    # x = last_desired_layer_from_original_model
    # x = Dense(512, activation="relu", name="xfer_dense_1")(x)
    # x = Dense(256, activation="relu", name="xfer_dense_2")(x)
    # x = Dense(128, activation="relu", name="xfer_dense_3")(x)
    # output_layer = Dense(1, activation="sigmoid", name="xfer_output")(x)
    #
    # transfer_learning_model = keras.models.Model(
    #     inputs=original_model.input, outputs=output_layer
    # )

    # Freeze the original model's layers up to the dense layers
    # the 9th index should be the flattening layer: "prep-for-dense"
    freeze_until_index = hyperparameters["freeze_until_index"]
    print("Freezing up to :", transfer_learning_model.layers[:freeze_until_index])

    # Freeze the original model's layers up to the dense layers
    for layer in transfer_learning_model.layers[:freeze_until_index]:
        layer.trainable = False

    # Allow the dense layers to be retrained (this is the default, so just adding this here to be explicit
    # TODO: probably remove this since it should be superfluous
    for layer in transfer_learning_model.layers[freeze_until_index:]:
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
    parser.add_argument("--original-model-id", type=str)
    parser.add_argument("--freeze-until-index", type=str)
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

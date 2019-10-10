"""
This model loads a pre-trained simple_cnn model and freezes some layers and retrains the rest
"""

import argparse
import sys

from cosmobot_deep_learning.analyze import load_model_from_h5
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
    # For using the Unit A scum + lab dataset
    # "dataset_filename": "2019-10-03--10-40-27_unit_A_scum_and_calibration.csv",
    #
    # For using the original Unit A scum-only dataset
    # "dataset_filename": "2019-09-26--14-19-22_osmo_ml_dataset_scum_dev_at_start.csv",
    # "training_set_column": "cosmobot_a_training_resampled",
    # "dev_set_column": "cosmobot_a_dev",
    #
    # For using the beta version of the Unit C scum + lab dataset (and hacking scum-only)
    # "dataset_filename": "2019-10-08--20-39-10_osmo_ml_dataset_unit_C_scum_and_calibration.csv",
    # HACK: temporary hack to get a scum-only dataset
    # "training_set_column": "scum_training",
    #
    # For using the beta version of trimmed Unit C scum-only
    "dataset_filename": "2019-10-09--20-55-53_osmo_ml_dataset_unit_C_scum_only.csv",
    "training_set_column": "training",
    #
    "dev_set_column": "dev_resampled",
    "numeric_input_columns": ["YSI temperature (C)"],
    "image_size": 128,
    # Use an even smaller learning rate than original model (default=0.0001)
    # so that we don't unlearn old things before we've learned new things
    # note: mixed success with this. adadelta optimizer seems to do better on higher learning rates
    "learning_rate": 0.00003,
    "freeze_until_index": 10,  # the 10th index should be the first dense layer ("dense_1")
    "original_model_id": "j53s5f07",  # The best baseline simple_cnn model so far
}


def create_model(hyperparameters, x_train):
    """ Build a model

    Args:
        hyperparameters: See definition in `run()`
        x_train: The input training data (used to determine input layer shape)
    """
    # Load a simple_cnn model that has been trained on calibration data
    # TODO: use wandb API to download this? (Current hack: I pre-downloaded it)
    model_filename = f'{hyperparameters["original_model_id"]}-model-best.h5'
    transfer_learning_model = load_model_from_h5(hyperparameters, model_filename)

    # Freeze the original model's layers up to the dense layers
    # The default is for layer.trainable=True, so all later layers will be trainable
    freeze_until_index = hyperparameters["freeze_until_index"]
    print("Freezing layers:", transfer_learning_model.layers[:freeze_until_index])
    for layer in transfer_learning_model.layers[:freeze_until_index]:
        layer.trainable = False

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
    parser.add_argument("--freeze-until-index", type=int)
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

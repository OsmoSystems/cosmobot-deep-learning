from typing import List

import keras

from cosmobot_deep_learning.constants import (
    ACCEPTABLE_FRACTION_OUTSIDE_ERROR,
    ACCEPTABLE_ERROR_MG_L,
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG,
    MG_L_TO_MMHG_AT_25_C_1_ATM,
)
from cosmobot_deep_learning.load_dataset import (
    get_pkg_dataset_filepath,
    get_dataset_csv_hash,
)

from cosmobot_deep_learning.custom_metrics import (
    get_fraction_outside_acceptable_error_fn,
)


def _guard_no_overridden_calculated_hyperparameters(calculated, model_specific):
    """ Don't allow the model to override hyperparameters that are calculated here
    """
    # Check for any shared keys
    overridden_parameters = set(calculated) & set(model_specific)

    if overridden_parameters:
        raise ValueError(
            "Model-specific hyperparameters attempting to override calculated parameters: {overridden_parameters}"
        )


def _calculate_additional_hyperparameters(
    dataset_filename, acceptable_error_mg_l, label_scale_factor_mmhg
):
    dataset_filepath = get_pkg_dataset_filepath(dataset_filename)
    dataset_hash = get_dataset_csv_hash(dataset_filepath)

    acceptable_error_mmhg = acceptable_error_mg_l * MG_L_TO_MMHG_AT_25_C_1_ATM

    # Ensure that our custom metric uses the same normalizing factor we use to scale our labels
    acceptable_error_normalized = acceptable_error_mmhg / label_scale_factor_mmhg
    fraction_outside_acceptable_error = get_fraction_outside_acceptable_error_fn(
        acceptable_error=acceptable_error_normalized
    )

    return {
        "dataset_filepath": dataset_filepath,
        "dataset_hash": dataset_hash,
        "acceptable_error_mmhg": acceptable_error_mmhg,
        "acceptable_error_normalized": acceptable_error_normalized,
        "metrics": [
            "mean_squared_error",
            "mean_absolute_error",
            fraction_outside_acceptable_error,
        ],
    }


DEFAULT_LABEL_COLUMN = "setpoint O2 (mmHg)"
DEFAULT_LOSS = "mean_squared_error"
DEFAULT_OPTIMIZER = keras.optimizers.Adadelta()
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAINING_SET_COLUMN = "training_resampled"
DEFAULT_DEV_SET_COLUMN = "test"


def get_hyperparameters(
    model_name: str,
    dataset_filename: str,
    numeric_input_columns: List[str],
    label_column: str = DEFAULT_LABEL_COLUMN,
    label_scale_factor_mmhg: float = ATMOSPHERIC_OXYGEN_PRESSURE_MMHG,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    optimizer=DEFAULT_OPTIMIZER,
    loss=DEFAULT_LOSS,
    acceptable_error_mg_l: float = ACCEPTABLE_ERROR_MG_L,
    acceptable_fraction_outside_error: float = ACCEPTABLE_FRACTION_OUTSIDE_ERROR,
    training_set_column: str = DEFAULT_TRAINING_SET_COLUMN,
    dev_set_column: str = DEFAULT_DEV_SET_COLUMN,
    dataset_cache_name: str = None,
    **model_specific_hyperparameters,
):
    """ This function:
        1) DRYs up the calculation of some hyperparameters, e.g. dataset_hash
        2) Provides a single location to define hyperparameters that we definitely want to share across models
        3) Guards that required hyperparameters have been defined

    Args:
        model_name: A string label for the model
        dataset_filename: Filename of the dataset to use for training
        numeric_input_columns: A List of column names from the dataset to use as numeric inputs (x) to the model
        label_column: A column name from the dataset to use as the labels (y) for the model
        label_scale_factor_mmhg: The scaling factor to use to scale labels into the [0,1] range
        epochs: Number of epochs to train for
        batch_size: Training batch size
        optimizer: Which optimizer function to use
        loss: Which loss function to use
        acceptable_error_mg_l: The threshold, in mg/L to use in our custom "fraction_outside_acceptable_error" metric
        acceptable_fraction_outside_error: The threshold fraction of predictions which
            can be outside the acceptable_error_mg_l
        training_set_column: The dataset column name of the training set flag.
        dev_set_column: The dataset column name of the dev set flag.
        **model_specific_hyperparameters: All other kwargs get slurped up here

    Returns: A dict of hyperparameters

    """
    calculated_hyperparameters = _calculate_additional_hyperparameters(
        dataset_filename, acceptable_error_mg_l, label_scale_factor_mmhg
    )

    _guard_no_overridden_calculated_hyperparameters(
        calculated_hyperparameters, model_specific_hyperparameters
    )

    return {
        # Pass through defined/default hyperparameters
        "model_name": model_name,
        "dataset_filename": dataset_filename,
        "dataset_cache_name": dataset_cache_name,
        "numeric_input_columns": numeric_input_columns,
        "label_column": label_column,
        "label_scale_factor_mmhg": label_scale_factor_mmhg,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss": loss,
        "acceptable_error_mg_l": acceptable_error_mg_l,
        "acceptable_fraction_outside_error": acceptable_fraction_outside_error,
        "training_set_column": training_set_column,
        "dev_set_column": dev_set_column,
        **calculated_hyperparameters,
        **model_specific_hyperparameters,
    }

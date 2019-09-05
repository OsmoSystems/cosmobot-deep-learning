import argparse
from typing import List, Set

from cosmobot_deep_learning.configure import parse_model_run_args
from cosmobot_deep_learning.constants import (
    ACCEPTABLE_FRACTION_OUTSIDE_ERROR,
    ACCEPTABLE_ERROR_MG_L,
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG,
    OPTIMIZER_CLASSES_BY_NAME,
)
from cosmobot_deep_learning.load_dataset import (
    get_pkg_dataset_filepath,
    get_dataset_csv_hash,
)

from cosmobot_deep_learning.custom_metrics import (
    get_fraction_outside_error_threshold_fn,
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
    dataset_filename,
    error_thresholds_mg_l: Set[float],
    acceptable_error_mg_l,
    label_scale_factor_mmhg,
):
    dataset_filepath = get_pkg_dataset_filepath(dataset_filename)
    dataset_hash = get_dataset_csv_hash(dataset_filepath)

    # Ensure acceptable_error_mg_l is always included in error_thresholds
    error_thresholds_mg_l = error_thresholds_mg_l.union({acceptable_error_mg_l})

    fraction_outside_error_threshold_fns = [
        get_fraction_outside_error_threshold_fn(
            error_threshold_mg_l, label_scale_factor_mmhg
        )
        for error_threshold_mg_l in error_thresholds_mg_l
    ]

    return {
        "dataset_filepath": dataset_filepath,
        "dataset_hash": dataset_hash,
        "metrics": ["mean_squared_error", "mean_absolute_error"]
        + fraction_outside_error_threshold_fns,
    }


LATEST_DATASET = "2019-09-04--17-21-54_osmo_ml_dataset.csv"
DEFAULT_LABEL_COLUMN = "setpoint O2 (mmHg)"
DEFAULT_LOSS = "mean_squared_error"
DEFAULT_OPTIMIZER_NAME = "adadelta"
DEFAULT_EPOCHS = 600
DEFAULT_BATCH_SIZE = 1000
DEFAULT_TRAINING_SET_COLUMN = "training_resampled"
DEFAULT_DEV_SET_COLUMN = "dev_resampled"
# Allow some breathing room when DO readings in mmHg go above atmospheric level
LABEL_SCALE_FACTOR_MMHG_BUFFER = 30
DEFAULT_LABEL_SCALE_FACTOR_MMHG = (
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG + LABEL_SCALE_FACTOR_MMHG_BUFFER
)
DEFAULT_ERROR_THRESHOLDS = {0.1, 0.3, 0.5}


def get_hyperparameters(
    model_name: str,
    numeric_input_columns: List[str],
    dataset_filename: str = LATEST_DATASET,
    label_column: str = DEFAULT_LABEL_COLUMN,
    label_scale_factor_mmhg: float = DEFAULT_LABEL_SCALE_FACTOR_MMHG,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    optimizer_name=DEFAULT_OPTIMIZER_NAME,
    loss=DEFAULT_LOSS,
    error_thresholds_mg_l: Set[float] = DEFAULT_ERROR_THRESHOLDS,
    acceptable_error_mg_l: float = ACCEPTABLE_ERROR_MG_L,
    acceptable_fraction_outside_error: float = ACCEPTABLE_FRACTION_OUTSIDE_ERROR,
    training_set_column: str = DEFAULT_TRAINING_SET_COLUMN,
    dev_set_column: str = DEFAULT_DEV_SET_COLUMN,
    dataset_cache_name: str = None,
    dryrun: bool = False,
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
        error_thresholds_mg_l: For each error threshold, compute the fraction of predictions that fall outside of it
        acceptable_error_mg_l: The threshold, in mg/L to use in our custom ThresholdValMeanAbsoluteErrorOnCustomMetric
        acceptable_fraction_outside_error: The threshold fraction of predictions which
            can be outside the acceptable_error_mg_l
        training_set_column: The dataset column name of the training set flag.
        dev_set_column: The dataset column name of the dev set flag.
        dryrun: Run on a tiny dataset for a single epoch and don't log to wandb.
        **model_specific_hyperparameters: All other kwargs get slurped up here

    Returns: A dict of hyperparameters
    """
    calculated_hyperparameters = _calculate_additional_hyperparameters(
        dataset_filename,
        error_thresholds_mg_l,
        acceptable_error_mg_l,
        label_scale_factor_mmhg,
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
        "optimizer_name": optimizer_name,
        "loss": loss,
        "acceptable_error_mg_l": acceptable_error_mg_l,
        "acceptable_fraction_outside_error": acceptable_fraction_outside_error,
        "training_set_column": training_set_column,
        "dev_set_column": dev_set_column,
        "dryrun": dryrun,
        **calculated_hyperparameters,
        **model_specific_hyperparameters,
    }


def _remove_items_with_no_value(dictionary):
    return {k: v for k, v in dictionary.items() if v is not None}


def get_hyperparameters_from_args(
    command_line_args: List[str],
    model_default_hyperparameters: dict,
    model_hyperparameter_parser: argparse.ArgumentParser = None,
):
    """Takes command line arguments, default hyperparameters for the model, and an optional
    command line parser for model-specific hyperparameters and returns a dict of
    hyperparameters with defaults overridden in the proper order.

    Args:
        command_line_args: list of command line args in the format returned by sys.argv[1:]
        model_default_hyperparameters: dict of model-specific hyperparameter defaults
        model_hyperparameter_parser: argparse.ArgumentParser with options for model-specific
        hyperparameters. add_help should be set to False.

    Returns: dict of hyperparameters
    """
    args = parse_model_run_args(command_line_args, model_hyperparameter_parser)

    # remove undefined arguments so that Nones don't override default values
    hyperparameters_from_args = _remove_items_with_no_value(vars(args))
    defaulted_hyperparameters = {
        **model_default_hyperparameters,
        **hyperparameters_from_args,
    }
    hyperparameters = get_hyperparameters(**defaulted_hyperparameters)

    return hyperparameters


def get_optimizer(hyperparameters):
    optimizer_name = hyperparameters["optimizer_name"]
    learning_rate = hyperparameters.get("learning_rate")

    optimizer_kwargs = {}
    if learning_rate is not None:
        optimizer_kwargs["lr"] = learning_rate

    optimizer_class = OPTIMIZER_CLASSES_BY_NAME[optimizer_name.lower()]

    return optimizer_class(**optimizer_kwargs)

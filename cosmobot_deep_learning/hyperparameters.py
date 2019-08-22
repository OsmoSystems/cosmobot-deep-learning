import keras

from cosmobot_deep_learning.constants import (
    ACCEPTABLE_ERROR_MG_L,
    ACCEPTABLE_ERROR_MMHG,
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG,
)
from cosmobot_deep_learning.load_dataset import (
    get_pkg_dataset_filepath,
    get_dataset_hash,
)

from cosmobot_deep_learning.custom_metrics import (
    get_fraction_outside_acceptable_error_fn,
)


DEFAULT_LABEL_COLUMN = "YSI DO (mmHg)"
DEFAULT_LOSS = "mean_squared_error"
DEFAULT_OPTIMIZER = keras.optimizers.Adadelta()
DEFAULT_EPOCHS = 10000
DEFAULT_BATCH_SIZE = 128


def get_hyperparameters(
    model_name,
    dataset_filename,
    input_columns,
    label_column=DEFAULT_LABEL_COLUMN,
    label_scale_factor_mmhg=ATMOSPHERIC_OXYGEN_PRESSURE_MMHG,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    optimizer=DEFAULT_OPTIMIZER,
    loss=DEFAULT_LOSS,
    acceptable_error_mg_l=ACCEPTABLE_ERROR_MG_L,
    acceptable_error_mmhg=ACCEPTABLE_ERROR_MMHG,
    **model_specific_hyperparameters,
):
    """ This function:
        1) DRYs up the calculation of some hyperparameters, e.g. dataset_hash
        2) Provides a single location to define hyperparameters that we definitely want to share across models
        3) Guards that required hyperparameters have been defined

    Args:
        TODO

    Returns: A dict of hyperparameters

    """
    dataset_filepath = get_pkg_dataset_filepath(dataset_filename)
    dataset_hash = get_dataset_hash(dataset_filepath)

    # Ensure that our custom metric uses the same normalizing factor we use to scale our labels
    acceptable_error_normalized = acceptable_error_mmhg / label_scale_factor_mmhg
    # TODO: pull out calculation of acceptable_error_mmhg from constants and don't pass it in

    fraction_outside_acceptable_error = get_fraction_outside_acceptable_error_fn(
        acceptable_error=acceptable_error_normalized
    )

    return {
        # Splat the model_specific_hyperparameters first so they can't override calculated ones
        # TODO: maybe add validation that this isn't happening
        **model_specific_hyperparameters,
        # Defined or default hyperparameters
        "model_name": model_name,
        "dataset_filename": dataset_filename,
        "input_columns": input_columns,
        "label_column": label_column,
        "label_scale_factor_mmhg": label_scale_factor_mmhg,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss": loss,
        "acceptable_error_mg_l": acceptable_error_mg_l,
        "acceptable_error_mmhg": acceptable_error_mmhg,
        # Cacluated hyperparameters
        "dataset_filepath": dataset_filepath,
        "dataset_hash": dataset_hash,
        "acceptable_error_normalized": acceptable_error_normalized,
        "metrics": [
            "mean_squared_error",
            "mean_absolute_error",
            fraction_outside_acceptable_error,
        ],
    }

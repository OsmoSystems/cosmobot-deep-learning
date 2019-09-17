import ast
from typing import Dict

import numpy as np
import pandas as pd


from cosmobot_deep_learning.preprocess_image import (
    open_and_preprocess_images,
    open_and_preprocess_image_ROIs,
)


SETPOINT_O2_FRACTION_COLUMN_NAME = "setpoint O2 (fraction)"
SETPOINT_TEMPERATURE_COLUMN_NAME = "setpoint temperature (C)"


def extract_inputs(df, input_column_names):
    """ Extract non-image input data values

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
            numeric_input_columns: A list of column names to be included as inputs.
                Special case: "sr" can be specified in this list to include the calculated spatial ratiometric value
        Returns:
            A numpy array of inputs, including just the values of the specified columns
    """
    # Manually add a spatial ratiometric "sr" column to the dataset, so that models can specify using it
    # as a numeric input simply by referring to it in the `numeric_input_columns` hyperparameter
    dataset = df.assign(
        **{"sr": df["DO patch r_msorm"] / df["reference patch r_msorm"]}
    )[input_column_names]

    return dataset.values


def extract_labels(df, label_column, label_scale_factor_mmhg):
    """ Get normalized label (y) data values for a given dataset (x)

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
            label_column: The column from the df to use as the labels
            label_scale_factor_mmhg: The scaling factor to use to scale labels into the [0,1] range
        Returns:
            Numpy array of dissolved oxygen label values, normalized by a constant scale factor
    """
    scaled_labels = df[label_column] / label_scale_factor_mmhg

    # Reshape to 2d array
    return np.reshape(scaled_labels.values, (-1, 1))


def prepare_dataset_numeric(raw_dataset: pd.DataFrame, hyperparameters):
    """ Transform a dataset CSV into the appropriate inputs and labels for training and
    validating a model.

        Args:
            raw_dataset: A DataFrame corresponding to a standard cosmobot dataset csv
            hyperparameters: A dictionary that includes at least:
                numeric_input_columns: A list of column names to be included as inputs.
                label_column: The column to use as the label (y) data values for a given dataset (x)
                label_scale_factor_mmhg: The scaling factor to use to scale labels into the [0,1] range
        Returns:
            A 4-tuple containing (x_train, y_train, x_dev, y_dev) data sets.
    """
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    dev_samples = raw_dataset[raw_dataset[dev_set_column]]

    x_train = extract_inputs(train_samples, numeric_input_columns)
    y_train = extract_labels(train_samples, label_column, label_scale_factor_mmhg)

    x_dev = extract_inputs(dev_samples, numeric_input_columns)
    y_dev = extract_labels(dev_samples, label_column, label_scale_factor_mmhg)

    return (x_train, y_train, x_dev, y_dev)


def _get_images_and_labels(
    raw_dataset: pd.DataFrame,
    sample_selector_column: str,
    image_size: int,
    label_column: str,
    label_scale_factor_mmhg: str,
):
    samples = raw_dataset[raw_dataset[sample_selector_column]]
    images = open_and_preprocess_images(samples["local_filepath"].values, image_size)
    labels = extract_labels(samples, label_column, label_scale_factor_mmhg)

    return images, labels


def prepare_dataset_image_only(raw_dataset: pd.DataFrame, hyperparameters: Dict):
    """ Transform a dataset CSV into the appropriate inputs and labels for training and
    validating a model, for a model that uses separate image and numeric inputs

        Args:
            raw_dataset: A DataFrame corresponding to a standard cosmobot dataset csv
            hyperparameters: A dictionary that includes at least:
                label_column: The column to use as the label (y) data values for a given dataset (x)
                label_scale_factor_mmhg: The scaling factor to use to scale labels into the [0,1] range
                image_size: The desired side length of the scaled (square) images
        Returns:
            A 4-tuple containing (x_train, y_train, x_dev, y_dev) data sets.
    """
    image_size = hyperparameters["image_size"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]

    x_train, y_train = _get_images_and_labels(
        raw_dataset,
        sample_selector_column=hyperparameters["training_set_column"],
        image_size=image_size,
        label_column=label_column,
        label_scale_factor_mmhg=label_scale_factor_mmhg,
    )
    x_dev, y_dev = _get_images_and_labels(
        raw_dataset,
        sample_selector_column=hyperparameters["dev_set_column"],
        image_size=image_size,
        label_column=label_column,
        label_scale_factor_mmhg=label_scale_factor_mmhg,
    )

    return (x_train, y_train, x_dev, y_dev)


def _uniformly_sample_array(arr: np.ndarray, output_length: int):
    sampling_indexes = np.linspace(
        start=0, stop=(len(arr) - 1), num=output_length, dtype=np.int16
    )
    return arr[sampling_indexes]


def _round_setpoint_columns(dataset):
    # round the setpoints in the dataset to get rid of floating point errors
    dataset[SETPOINT_O2_FRACTION_COLUMN_NAME] = dataset[
        SETPOINT_O2_FRACTION_COLUMN_NAME
    ].round(decimals=5)
    dataset[SETPOINT_TEMPERATURE_COLUMN_NAME] = dataset[
        SETPOINT_TEMPERATURE_COLUMN_NAME
    ].round(decimals=3)


def _downsample_setpoints(dataset, setpoint_column_name, desired_num_setpoints):
    setpoint_values = dataset[setpoint_column_name].unique().sort()

    print(f"{setpoint_column_name} values: {setpoint_values}")

    num_setpoints = len(setpoint_values)

    if num_setpoints < desired_num_setpoints:
        raise Exception(
            f"too many setpoints requested ({len(setpoint_values)} in dataset, {desired_num_setpoints} requested)"
        )

    print(
        f'reducing setpoints for "{setpoint_column_name}" from {len(setpoint_values)} to {desired_num_setpoints}'
    )

    # uniformly sample num_setpoints values from the list of unique setpoint values
    sampled_setpoint_values = _uniformly_sample_array(
        setpoint_values, desired_num_setpoints
    )
    print(f"{setpoint_column_name} downsampled values: {sampled_setpoint_values}")

    # return raw dataset filtered down to those setpoints
    sampled_dataset = dataset[
        dataset[setpoint_column_name].isin(sampled_setpoint_values)
    ]
    return sampled_dataset


def _downsample_dataset(dataset, desired_size):
    """Return a copy of the dataset downsampled to the desired size (rows).

    This will sort the values by [SETPOINT_TEMPERATURE_COLUMN_NAME, SETPOINT_O2_FRACTION_COLUMN_NAME] to
    make the sampling somewhat uniform across setpoints.
    """
    dataset_size = len(dataset)

    if desired_size > dataset_size:
        raise Exception(
            f"trying to downsample dataset of size {dataset_size} to larger size of {desired_size}. "
            "you're probably doing something wrong  ಠ_ಠ "
        )

    print(f"downsampling dataset of size {dataset_size} to {desired_size}")

    sampling_indexes = np.linspace(
        start=0, stop=(dataset_size - 1), num=desired_size, dtype=np.int16
    )

    downsampled_dataset = (
        dataset.sort_values(
            [SETPOINT_TEMPERATURE_COLUMN_NAME, SETPOINT_O2_FRACTION_COLUMN_NAME]
        )
        .reset_index(drop=True)
        .iloc[sampling_indexes]
    )

    assert len(downsampled_dataset) == desired_size
    print(f"downsampled dataset size: {len(downsampled_dataset)}")

    return downsampled_dataset


def _reduce_images_per_setpoint(dataset, max_images_per_setpoint):
    """Return dataset with at most max_images_per_setpoint images per setpoint.
    """
    first_image_indexes = (
        dataset.reset_index()  # move old indexes to an 'index' column
        .groupby(
            [SETPOINT_TEMPERATURE_COLUMN_NAME, SETPOINT_O2_FRACTION_COLUMN_NAME]
        )  # sorted by default
        .nth(list(range(max_images_per_setpoint)))[
            "index"
        ]  # get indexes of first item in each group
        .tolist()
    )
    return dataset.loc[first_image_indexes]


def prepare_dataset_image_and_numeric(raw_dataset: pd.DataFrame, hyperparameters):
    """ Transform a dataset CSV into the appropriate inputs and labels for training and
    validating a model, for a model that uses separate image and numeric inputs

        Args:
            raw_dataset: A DataFrame corresponding to a standard cosmobot dataset csv
            hyperparameters: A dictionary that includes at least:
                numeric_input_columns: A list of column names to be included as inputs.
                label_column: The column to use as the label (y) data values for a given dataset (x)
                label_scale_factor_mmhg: The scaling factor to use to scale labels into the [0,1] range
                image_size: The desired side length of the scaled (square) images
        Returns:
            A 4-tuple containing (x_train, y_train, x_dev, y_dev) data sets.
    """

    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    image_size = hyperparameters["image_size"]
    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]

    # clean up floating point discrepencies for downsampling number of setpoint values
    _round_setpoint_columns(raw_dataset)

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    dev_samples = raw_dataset[raw_dataset[dev_set_column]]

    print(f"original train sample count: {len(train_samples)}")

    if hyperparameters.get("num_do_setpoints") is not None:
        train_samples = _downsample_setpoints(
            train_samples,
            SETPOINT_O2_FRACTION_COLUMN_NAME,
            hyperparameters["num_do_setpoints"],
        )

    if hyperparameters.get("num_temp_setpoints") is not None:
        train_samples = _downsample_setpoints(
            train_samples,
            SETPOINT_TEMPERATURE_COLUMN_NAME,
            hyperparameters["num_temp_setpoints"],
        )

    max_images_per_setpoint = hyperparameters.get("max_images_per_setpoint")
    if max_images_per_setpoint:
        train_samples = _reduce_images_per_setpoint(
            train_samples, max_images_per_setpoint
        )

    # TODO ability to reduce number of replicates? (I don't know if we have a way to do that in here)

    # this filter should be performed last
    if hyperparameters.get("train_sample_count") is not None:
        train_samples = _downsample_dataset(
            train_samples, hyperparameters["train_sample_count"]
        )

    print(f"filtered train sample count: {len(train_samples)}")

    x_train_numeric = extract_inputs(train_samples, numeric_input_columns)
    x_train_images = open_and_preprocess_images(
        train_samples["local_filepath"].values, image_size
    )
    y_train = extract_labels(train_samples, label_column, label_scale_factor_mmhg)

    x_dev_numeric = extract_inputs(dev_samples, numeric_input_columns)
    x_dev_images = open_and_preprocess_images(
        dev_samples["local_filepath"].values, image_size
    )
    y_dev = extract_labels(dev_samples, label_column, label_scale_factor_mmhg)

    return (
        [x_train_numeric, x_train_images],
        y_train,
        [x_dev_numeric, x_dev_images],
        y_dev,
    )


def _extract_ROI_and_numeric_features_and_labels(samples, hyperparameters):
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    image_size = hyperparameters["image_size"]
    input_ROI_names = hyperparameters["input_ROI_names"]

    # Use ast to safely eval ROI definitions to dict
    ROI_definitions = samples["ROI definitions"].apply(ast.literal_eval)

    x_numeric = extract_inputs(samples, numeric_input_columns)
    x_crops = open_and_preprocess_image_ROIs(
        list(zip(samples["local_filepath"], ROI_definitions)),
        input_ROI_names,
        image_size,
    )
    y = extract_labels(samples, label_column, label_scale_factor_mmhg)

    return ([x_numeric] + list(x_crops), y)


def prepare_dataset_ROIs_and_numeric(raw_dataset: pd.DataFrame, hyperparameters):
    """ Transform a dataset CSV into the appropriate inputs and labels for training and
    validating a model, for a model that uses multiple ROI-crop input images as well as numeric inputs

        Args:
            raw_dataset: A DataFrame corresponding to a standard cosmobot dataset csv
            hyperparameters: A dictionary that includes at least:
                numeric_input_columns: A list of column names to be included as inputs.
                label_column: The column to use as the label (y) data values for a given dataset (x)
                label_scale_factor_mmhg: The scaling factor to use to scale labels into the [0,1] range
                image_size: The desired side length of the scaled (square) images
                input_ROI_names: The names of ROIs to extract from images as model inputs
        Returns:
            A 4-tuple containing (x_train, y_train, x_dev, y_dev) data sets.
    """
    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    dev_samples = raw_dataset[raw_dataset[dev_set_column]]

    x_train, y_train = _extract_ROI_and_numeric_features_and_labels(
        train_samples, hyperparameters
    )
    x_dev, y_dev = _extract_ROI_and_numeric_features_and_labels(
        dev_samples, hyperparameters
    )

    return (x_train, y_train, x_dev, y_dev)

import ast
from typing import Dict

import numpy as np
import pandas as pd


from cosmobot_deep_learning.preprocess_image import (
    open_and_preprocess_images,
    open_and_preprocess_image_ROIs,
)


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


def _extract_image_and_numeric_features_and_labels(samples, hyperparameters):
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    image_size = hyperparameters["image_size"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]

    x_numeric = extract_inputs(samples, numeric_input_columns)
    x_images = open_and_preprocess_images(samples["local_filepath"].values, image_size)
    y = extract_labels(samples, label_column, label_scale_factor_mmhg)

    return ([x_numeric, x_images], y)


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
    # HACK: add in a column for a scum-only training set
    scum_samples = raw_dataset["scum tank"]
    all_training_samples = raw_dataset["training"]
    raw_dataset["scum_training"] = raw_dataset[scum_samples & all_training_samples]
    # END HACK

    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    dev_samples = raw_dataset[raw_dataset[dev_set_column]]

    x_train, y_train = _extract_image_and_numeric_features_and_labels(
        train_samples, hyperparameters
    )

    x_dev, y_dev = _extract_image_and_numeric_features_and_labels(
        dev_samples, hyperparameters
    )

    return (x_train, y_train, x_dev, y_dev)


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

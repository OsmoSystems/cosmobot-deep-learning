import ast
from typing import Dict

import numpy as np
import pandas as pd


from cosmobot_deep_learning.preprocess_image import (
    open_and_preprocess_images,
    open_and_preprocess_image_ROIs,
)


def extract_numeric_inputs(df, input_column_names):
    """ Extract non-image input data values

        Args:
            df: A DataFrame representing a standard cosmobot dataset (from /datasets)
            numeric_input_columns: A list of column names to be included as inputs.
        Returns:
            A numpy array of inputs, including just the values of the specified columns
    """

    return df[input_column_names].values


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
            A 2-tuple containing (x, y) data sets.
    """
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]

    x = extract_numeric_inputs(raw_dataset, numeric_input_columns)
    y = extract_labels(raw_dataset, label_column, label_scale_factor_mmhg)

    return (x, y)


def _get_images_and_labels(
    samples: pd.DataFrame,
    image_size: int,
    label_column: str,
    label_scale_factor_mmhg: str,
):
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
            A 2-tuple containing (x, y) data sets.
    """
    image_size = hyperparameters["image_size"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]

    x, y = _get_images_and_labels(
        raw_dataset,
        image_size=image_size,
        label_column=label_column,
        label_scale_factor_mmhg=label_scale_factor_mmhg,
    )

    return (x, y)


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
            A 2-tuple containing (x, y) data sets.
    """
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    image_size = hyperparameters["image_size"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]

    x_numeric = extract_numeric_inputs(raw_dataset, numeric_input_columns)

    x_images = open_and_preprocess_images(
        raw_dataset["local_filepath"].values, image_size
    )
    y = extract_labels(raw_dataset, label_column, label_scale_factor_mmhg)

    return ([x_numeric, x_images], y)


def _extract_ROI_and_numeric_features_and_labels(samples, hyperparameters):
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    image_size = hyperparameters["image_size"]
    input_ROI_names = hyperparameters["input_ROI_names"]

    # Use ast to safely eval ROI definitions to dict
    ROI_definitions = samples["ROI definitions"].apply(ast.literal_eval)

    x_numeric = extract_numeric_inputs(samples, numeric_input_columns)
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
            A 2-tuple containing (x, y) data sets.
    """
    x, y = _extract_ROI_and_numeric_features_and_labels(raw_dataset, hyperparameters)

    return (x, y)

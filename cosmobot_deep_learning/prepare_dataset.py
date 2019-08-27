import ast

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
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    test_samples = raw_dataset[raw_dataset[dev_set_column]]

    x_train = extract_inputs(train_samples, numeric_input_columns)
    y_train = extract_labels(train_samples, label_column, label_scale_factor_mmhg)

    x_test = extract_inputs(test_samples, numeric_input_columns)
    y_test = extract_labels(test_samples, label_column, label_scale_factor_mmhg)

    return (x_train, y_train, x_test, y_test)


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
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    image_size = hyperparameters["image_size"]
    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    test_samples = raw_dataset[raw_dataset[dev_set_column]]

    x_train_numeric = extract_inputs(train_samples, numeric_input_columns)
    x_train_images = open_and_preprocess_images(
        train_samples["local_filepath"].values, image_size
    )
    y_train = extract_labels(train_samples, label_column, label_scale_factor_mmhg)

    x_test_numeric = extract_inputs(test_samples, numeric_input_columns)
    x_test_images = open_and_preprocess_images(
        test_samples["local_filepath"].values, image_size
    )
    y_test = extract_labels(test_samples, label_column, label_scale_factor_mmhg)

    return (
        [x_train_numeric, x_train_images],
        y_train,
        [x_test_numeric, x_test_images],
        y_test,
    )


def prepare_dataset_ROIs_and_numeric(raw_dataset: pd.DataFrame, hyperparameters):
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
            A 4-tuple containing (x_train, y_train, x_test, y_test) data sets.
    """
    numeric_input_columns = hyperparameters["numeric_input_columns"]
    label_column = hyperparameters["label_column"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    image_size = hyperparameters["image_size"]
    training_set_column = hyperparameters["training_set_column"]
    dev_set_column = hyperparameters["dev_set_column"]
    image_input_ids = hyperparameters["image_input_ids"]

    train_samples = raw_dataset[raw_dataset[training_set_column]]
    test_samples = raw_dataset[raw_dataset[dev_set_column]]

    # Use ast to safely eval ROI definitions to dict
    train_ROI_definitions = train_samples["ROI definitions"].apply(ast.literal_eval)

    x_train_numeric = extract_inputs(train_samples, numeric_input_columns)
    x_train_crops = open_and_preprocess_image_ROIs(
        list(zip(train_samples["local_filepath"], train_ROI_definitions)),
        image_input_ids,
        image_size,
    )
    y_train = extract_labels(train_samples, label_column, label_scale_factor_mmhg)

    test_ROI_definitions = test_samples["ROI definitions"].apply(ast.literal_eval)

    x_test_numeric = extract_inputs(test_samples, numeric_input_columns)
    x_test_crops = open_and_preprocess_image_ROIs(
        list(zip(test_samples["local_filepath"], test_ROI_definitions)),
        image_input_ids,
        image_size,
    )
    y_test = extract_labels(test_samples, label_column, label_scale_factor_mmhg)

    return (
        [x_train_numeric] + list(x_train_crops),
        y_train,
        [x_test_numeric] + list(x_test_crops),
        y_test,
    )

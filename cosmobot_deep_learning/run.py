import os
import logging
import pickle

import pandas as pd
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import (
    get_dataset_cache_filepath,
    download_images_and_attach_filepaths_to_dataset,
    get_loaded_dataset_hash,
)

from cosmobot_deep_learning.custom_metrics import (
    magical_incantation_to_make_custom_metric_work,
)
from cosmobot_deep_learning import visualizations


def _loggable_hyperparameters(hyperparameters):
    # W&B logging chokes on our custom metric function.
    # Manually fix this by replacing metric function with its __name__
    loggable_metrics = [
        metric.__name__ if hasattr(metric, "__name__") else metric
        for metric in hyperparameters["metrics"]
    ]

    return {
        **hyperparameters,
        # Override the original "unloggable" metrics key
        "metrics": loggable_metrics,
    }


def _initialize_wandb(hyperparameters):
    wandb.init(
        entity="osmo",
        project="cosmobot-do-measurement",
        config={**_loggable_hyperparameters(hyperparameters)},
    )


def _update_wandb_with_loaded_dataset(loaded_dataset):
    """ update wandb configuration hyperparameters with information about the dataset that's been loaded

    Args:
        loaded_dataset: (x_train, y_train, x_test, y_test) tuple of data being used for modeling

    Returns:
        None
    """
    loaded_dataset_hash = get_loaded_dataset_hash(loaded_dataset)

    x_train, y_train, x_test, y_test = loaded_dataset
    wandb.config.update(
        {
            "loaded_dataset_hash": loaded_dataset_hash,
            "train_sample_count": y_train.shape[0],
            "test_sample_count": y_test.shape[0],
        }
    )


def _log_visualizations(
    model, training_history, label_scale_factor_mmhg, x_train, y_train, x_test, y_test
):
    train_labels = y_train.flatten() * label_scale_factor_mmhg
    train_predictions = model.predict(x_train).flatten() * label_scale_factor_mmhg

    dev_labels = y_test.flatten() * label_scale_factor_mmhg
    dev_predictions = model.predict(x_test).flatten() * label_scale_factor_mmhg

    visualizations.log_loss_over_epochs(training_history)
    visualizations.log_do_prediction_error(
        train_labels, train_predictions, dev_labels, dev_predictions
    )
    visualizations.log_actual_vs_predicted_do(
        train_labels, train_predictions, dev_labels, dev_predictions
    )


def _generate_tiny_dataset(dataset, hyperparameters):
    """ Grab the first two training and dev data points to create a tiny dataset.
    """
    training_sample = dataset[dataset[hyperparameters["training_set_column"]]][:2]
    test_sample = dataset[dataset[hyperparameters["dev_set_column"]]][:2]
    return training_sample.append(test_sample)


def _shuffle_dataframe(dataframe):
    return dataframe.sample(
        n=len(dataframe),  # sample all rows, essentially shuffling
        random_state=0,  # set a constant seed for consistent shuffling
    ).reset_index(
        drop=True
    )  # reset index to match new order, and drop the old index values


def _get_prepared_dataset(prepare_dataset, hyperparameters, dryrun):
    dataset_filepath = hyperparameters["dataset_filepath"]
    dataset = pd.read_csv(dataset_filepath)

    if dryrun:
        dataset = _generate_tiny_dataset(dataset, hyperparameters)

    shuffled_dataset = _shuffle_dataframe(dataset)

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=download_images_and_attach_filepaths_to_dataset(shuffled_dataset),
        hyperparameters=hyperparameters,
    )

    return x_train, y_train, x_test, y_test


def _load_dataset_cache(dataset_cache_filepath):
    with open(dataset_cache_filepath, "rb") as cache_file:
        return pickle.load(cache_file)


def _save_dataset_cache(dataset_cache_filepath, dataset):
    with open(dataset_cache_filepath, "wb+") as cache_file:
        pickle.dump(dataset, cache_file)


def _prepare_dataset_with_caching(
    prepare_dataset, hyperparameters, dryrun, dataset_cache_name=None
):
    dataset_cache_filepath = get_dataset_cache_filepath(dataset_cache_name)

    # Early exit with the cached datatset if it exists
    if dataset_cache_name is not None and os.path.isfile(dataset_cache_filepath):
        logging.info(f"Using dataset cache file {dataset_cache_filepath}")

        return _load_dataset_cache(dataset_cache_filepath)

    else:
        logging.info(f"Preparing dataset")

        dataset = _get_prepared_dataset(
            prepare_dataset=prepare_dataset,
            hyperparameters=hyperparameters,
            dryrun=dryrun,
        )

        if dataset_cache_name is not None:
            logging.info(f"Creating new dataset cache file {dataset_cache_name}")

            _save_dataset_cache(dataset_cache_filepath, dataset)

        return dataset


def _slice_all_arrays_in_list(maybe_list_of_arrays, slice_size):
    # Sometimes x input is a list of arrays (e.g. [x_train_numeric, x_train_images])
    # sometimes just one array (e.g. x_train_numeric)
    if type(maybe_list_of_arrays) == list:
        return [array[:slice_size] for array in maybe_list_of_arrays]

    return maybe_list_of_arrays[:slice_size]


def _downsample_training_dataset(loaded_dataset, train_sample_count):
    x_train, y_train, x_test, y_test = loaded_dataset

    return (
        _slice_all_arrays_in_list(x_train, train_sample_count),
        _slice_all_arrays_in_list(y_train, train_sample_count),
        x_test,
        y_test,
    )


def run(
    hyperparameters,
    prepare_dataset,
    create_model,
    dryrun=False,
    dataset_cache_name=None,
):
    """ Use the provided hyperparameters to train the model in this module.

    Args:
        hyperparameters: Any variables that are parameterizable for this model. See `get_hyperparameters` for details
        prepare_dataset: A function that takes a raw_dataset and returns (x_train, y_train, x_test, y_test)
        create_model: A function that takes hyperparameters and x_train and returns a compiled model
        dryrun: Whether the model should be run with a tiny dataset in dryrun mode.
        dataset_cache_name: Optional. Name of dataset cache file to load from for this run or save to for future runs.
    """
    logging_format = "%(asctime)s [%(levelname)s]--- %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=logging_format, handlers=[logging.StreamHandler()]
    )

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]

    if dryrun:
        epochs = 1
        # Disable W&B syncing to the cloud since we don't care about the results
        # os.environ["WANDB_MODE"] = "dryrun"

    _initialize_wandb(hyperparameters=hyperparameters)

    loaded_dataset = _prepare_dataset_with_caching(
        prepare_dataset=prepare_dataset,
        hyperparameters=hyperparameters,
        dryrun=dryrun,
        dataset_cache_name=dataset_cache_name,
    )

    # Temporary: downsample training dataset to a desired size
    downsampled_loaded_dataset = _downsample_training_dataset(
        loaded_dataset, hyperparameters["train_sample_count"]
    )

    _update_wandb_with_loaded_dataset(downsampled_loaded_dataset)

    x_train, y_train, x_test, y_test = downsampled_loaded_dataset

    model = create_model(hyperparameters, x_train)

    magical_incantation_to_make_custom_metric_work()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_test, y_test),
        callbacks=[WandbCallback()],
    )

    _log_visualizations(
        model,
        history,
        hyperparameters["label_scale_factor_mmhg"],
        x_train,
        y_train,
        x_test,
        y_test,
    )

    return x_train, y_train, x_test, y_test, model, history

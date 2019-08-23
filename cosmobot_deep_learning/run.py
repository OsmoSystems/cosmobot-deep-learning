import os
import logging
import pickle

import pandas as pd
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import load_multi_experiment_dataset_csv

from cosmobot_deep_learning.custom_metrics import (
    magical_incantation_to_make_custom_metric_work,
)


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


def _initialize_wandb(hyperparameters, y_train, y_test):
    wandb.init(
        entity="osmo",
        project="cosmobot-do-measurement",
        config={
            **_loggable_hyperparameters(hyperparameters),
            "train_sample_count": y_train.shape[0],
            "test_sample_count": y_test.shape[0],
        },
    )


def _generate_tiny_dataset(dataset, hyperparameters):
    """ Grab the first two training and dev data points to create a tiny dataset.
    """
    training_sample = dataset[dataset[hyperparameters["training_set_column"]]][:2]
    test_sample = dataset[dataset[hyperparameters["dev_set_column"]]][:2]
    return training_sample.append(test_sample)


def _prepare_dataset_with_caching(
    raw_dataset, prepare_dataset, hyperparameters, dataset_cache_name, cache_directory
):
    if dataset_cache_name is not None:
        cache_filepath = os.path.join(cache_directory, f"{dataset_cache_name}.pickle")

        if os.path.isfile(cache_filepath):
            with open(cache_filepath, "rb") as cache_file:
                return pickle.load(cache_file)

        logging.info(
            f"Dataset cache file {dataset_cache_name}.pickle does not exist - preparing dataset"
        )

    prepared_dataset = prepare_dataset(
        raw_dataset=raw_dataset, hyperparameters=hyperparameters
    )

    if dataset_cache_name is not None:
        logging.info(f"Saving prepared dataset as {dataset_cache_name}.pickle")
        with open(cache_filepath, "wb+") as cache_file:
            pickle.dump(prepared_dataset, cache_file)

    return prepared_dataset


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
    dataset_filepath = hyperparameters["dataset_filepath"]

    dataset = pd.read_csv(dataset_filepath)

    if dryrun:
        epochs = 1
        dataset = _generate_tiny_dataset(dataset, hyperparameters)
        # Disable W&B syncing to the cloud since we don't care about the results
        os.environ["WANDB_MODE"] = "dryrun"

    x_train, y_train, x_test, y_test = _prepare_dataset_with_caching(
        raw_dataset=load_multi_experiment_dataset_csv(dataset),
        prepare_dataset=prepare_dataset,
        hyperparameters=hyperparameters,
        dataset_cache_name=dataset_cache_name,
        cache_directory=os.path.expanduser("~/osmo/data-set-cache"),
    )

    _initialize_wandb(hyperparameters, y_train, y_test)

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

    return x_train, y_train, x_test, y_test, model, history

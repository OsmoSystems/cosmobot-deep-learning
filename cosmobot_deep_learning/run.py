import os
import io
import logging
import pickle
import subprocess

import pandas as pd
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.constants import LARGE_FILE_PICKLE_PROTOCOL
from cosmobot_deep_learning.load_dataset import (
    get_dataset_cache_filepath,
    download_images_and_attach_filepaths_to_dataset,
    get_loaded_dataset_hash,
)

from cosmobot_deep_learning.custom_metrics import (
    ThresholdValMeanAbsoluteErrorOnCustomMetric,
    magical_incantation_to_make_custom_metric_work,
    ErrorAtPercentile,
)
from cosmobot_deep_learning import visualizations

GPU_AVAILABLE_MEMORY_THRESHOLD = 7000


class NoGPUAvailable(Exception):
    # Raised when no GPUs are available for training and --no-gpu or --dryrun were not set
    pass


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
    wandb.init(config={**_loggable_hyperparameters(hyperparameters)})


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


def _get_prepared_dataset(prepare_dataset, hyperparameters):
    dataset_filepath = hyperparameters["dataset_filepath"]
    dataset = pd.read_csv(dataset_filepath)

    if hyperparameters["dryrun"]:
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
        pickle.dump(dataset, cache_file, protocol=LARGE_FILE_PICKLE_PROTOCOL)


def _prepare_dataset_with_caching(prepare_dataset, hyperparameters):
    dataset_cache_name = hyperparameters["dataset_cache_name"]
    dataset_cache_filepath = get_dataset_cache_filepath(dataset_cache_name)

    # Early exit with the cached datatset if it exists
    if dataset_cache_name is not None and os.path.isfile(dataset_cache_filepath):
        logging.info(f"Using dataset cache file {dataset_cache_filepath}")

        return _load_dataset_cache(dataset_cache_filepath)

    else:
        logging.info(f"Preparing dataset")

        dataset = _get_prepared_dataset(
            prepare_dataset=prepare_dataset, hyperparameters=hyperparameters
        )

        if dataset_cache_name is not None:
            logging.info(f"Creating new dataset cache file {dataset_cache_name}")

            _save_dataset_cache(dataset_cache_filepath, dataset)

        return dataset


def _get_gpu_stats():
    gpu_stats = pd.read_csv(
        io.StringIO(
            subprocess.check_output(
                'nvidia-smi --query-gpu="index,memory.free" --format=csv', shell=True
            ).decode("utf-8")
        ),
        header=0,
        names=["GPU ID", "Memory Free (MiB)"],
    )

    gpu_stats["Memory Free (MiB)"] = (
        gpu_stats["Memory Free (MiB)"].str.extract(r"(\d+)").astype(int)
    )

    return gpu_stats


def _set_cuda_visible_devices(no_gpu, dryrun):
    """ Assign CUDA_VISIBLE_DEVICES to any available GPU.
        If no_gpu or dryrun are truthy, set CUDA_VISIBLE_DEVICES to -1 for CPU training.
    """

    if no_gpu or dryrun:
        logging.info("Setting CUDA_VISIBLE_DEVICES to -1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    else:
        gpu_stats = _get_gpu_stats()

        # Get devices where most of the memory is free
        free_gpus = gpu_stats[
            gpu_stats["Memory Free (MiB)"] > GPU_AVAILABLE_MEMORY_THRESHOLD
        ]

        if not free_gpus["GPU ID"].size:
            raise NoGPUAvailable("No GPUs with enough available memory")

        device_id = free_gpus["GPU ID"].iloc[0]

        logging.info(f"Setting CUDA_VISIBLE_DEVICES to {device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def run(hyperparameters, prepare_dataset, create_model):
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
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    acceptable_error_mg_l = hyperparameters["acceptable_error_mg_l"]
    acceptable_fraction_outside_error = hyperparameters[
        "acceptable_fraction_outside_error"
    ]

    if hyperparameters["dryrun"]:
        epochs = 1
        # Disable W&B syncing to the cloud since we don't care about the results
        os.environ["WANDB_MODE"] = "dryrun"

    _initialize_wandb(hyperparameters=hyperparameters)

    loaded_dataset = _prepare_dataset_with_caching(
        prepare_dataset=prepare_dataset, hyperparameters=hyperparameters
    )

    _update_wandb_with_loaded_dataset(loaded_dataset)

    x_train, y_train, x_test, y_test = loaded_dataset

    _set_cuda_visible_devices(hyperparameters["no_gpu"], hyperparameters["dryrun"])

    model = create_model(hyperparameters, x_train)

    magical_incantation_to_make_custom_metric_work()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_test, y_test),
        callbacks=[
            ErrorAtPercentile(
                percentile=95,
                label_scale_factor_mmhg=label_scale_factor_mmhg,
                dataset=loaded_dataset,
            ),
            ThresholdValMeanAbsoluteErrorOnCustomMetric(
                acceptable_fraction_outside_error=acceptable_fraction_outside_error,
                acceptable_error_mg_l=acceptable_error_mg_l,
            ),
            WandbCallback(verbose=1, monitor="val_adjusted_mean_absolute_error"),
        ],
    )

    _log_visualizations(
        model, history, label_scale_factor_mmhg, x_train, y_train, x_test, y_test
    )

    return x_train, y_train, x_test, y_test, model, history

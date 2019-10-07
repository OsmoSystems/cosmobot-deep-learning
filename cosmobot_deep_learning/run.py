import os
import logging
import pickle

import pandas as pd
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.constants import LARGE_FILE_PICKLE_PROTOCOL
from cosmobot_deep_learning.custom_metrics import (
    ThresholdValMeanAbsoluteErrorOnCustomMetric,
    ErrorAtPercentile,
    RestoreBestWeights,
    SaveBestMetricValueAndEpochToWandb,
)
from cosmobot_deep_learning.gpu import (
    set_cuda_visible_devices,
    dont_use_all_the_gpu_memory,
)
from cosmobot_deep_learning.load_dataset import (
    get_dataset_cache_filepath,
    download_images_and_attach_filepaths_to_dataset,
    get_loaded_dataset_hash,
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
    wandb.init(config={**_loggable_hyperparameters(hyperparameters)})


def _update_wandb_with_loaded_dataset(loaded_dataset):
    """ update wandb configuration hyperparameters with information about the dataset that's been loaded

    Args:
        loaded_dataset: (x_train, y_train, x_dev, y_dev) tuple of data being used for modeling

    Returns:
        None
    """
    loaded_dataset_hash = get_loaded_dataset_hash(loaded_dataset)

    x_train, y_train, x_dev, y_dev = loaded_dataset
    wandb.config.update(
        {
            "loaded_dataset_hash": loaded_dataset_hash,
            "train_sample_count": y_train.shape[0],
            "dev_sample_count": y_dev.shape[0],
        }
    )


def _log_visualizations(
    model, training_history, label_scale_factor_mmhg, x_train, y_train, x_dev, y_dev
):
    train_labels = y_train.flatten() * label_scale_factor_mmhg
    train_predictions = model.predict(x_train).flatten() * label_scale_factor_mmhg

    dev_labels = y_dev.flatten() * label_scale_factor_mmhg
    dev_predictions = model.predict(x_dev).flatten() * label_scale_factor_mmhg

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
    dev_sample = dataset[dataset[hyperparameters["dev_set_column"]]][:2]
    return training_sample.append(dev_sample)


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

    x_train, y_train, x_dev, y_dev = prepare_dataset(
        raw_dataset=download_images_and_attach_filepaths_to_dataset(shuffled_dataset),
        hyperparameters=hyperparameters,
    )

    return x_train, y_train, x_dev, y_dev


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


def run(hyperparameters, prepare_dataset, create_model):
    """ Use the provided hyperparameters to train the model in this module.

    Args:
        hyperparameters: Any variables that are parameterizable for this model. See `get_hyperparameters` for details
        prepare_dataset: A function that takes a raw_dataset and returns (x_train, y_train, x_dev, y_dev)
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

    x_train, y_train, x_dev, y_dev = loaded_dataset

    set_cuda_visible_devices(hyperparameters["gpu"])
    dont_use_all_the_gpu_memory()

    wandb.config.update({"CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES")})

    model = create_model(hyperparameters, x_train)

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(x_dev, y_dev),
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
            SaveBestMetricValueAndEpochToWandb(
                metric="val_adjusted_mean_absolute_error"
            ),
            WandbCallback(verbose=1, monitor="val_adjusted_mean_absolute_error"),
            RestoreBestWeights(metric="val_adjusted_mean_absolute_error"),
        ],
    )

    _log_visualizations(
        model, history, label_scale_factor_mmhg, x_train, y_train, x_dev, y_dev
    )

    return x_train, y_train, x_dev, y_dev, model, history

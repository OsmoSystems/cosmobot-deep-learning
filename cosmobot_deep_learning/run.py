import os

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


def run(hyperparameters, prepare_dataset, create_model):
    """ Use the provided hyperparameters to train the model in this module.

    Args:
        hyperparameters: Any variables that are parameterizable for this model. See `get_hyperparameters` for details
        prepare_dataset: A function that takes a raw_dataset and returns (x_train, y_train, x_test, y_test)
        create_model: A function that takes hyperparameters and x_train and returns a compiled model

    """

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    dataset_filepath = hyperparameters["dataset_filepath"]
    tiny = hyperparameters["tiny"]

    dataset = pd.read_csv(dataset_filepath)

    if tiny:
        # Grab the first training and dev data points and replace the existing dataset
        dataset = dataset[dataset["training_resampled"]][:1].append(
            dataset[dataset["test"]][:1]
        )
        # Disable W&B logging since we don't care about the results
        os.environ["WANDB_MODE"] = "dryrun"

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=load_multi_experiment_dataset_csv(dataset),
        hyperparameters=hyperparameters,
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

import os

import pandas as pd
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import load_multi_experiment_dataset_csv
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
        # Disable W&B syncing to the cloud since we don't care about the results
        os.environ["WANDB_MODE"] = "dryrun"

    shuffled_dataset = _shuffle_dataframe(dataset)

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=load_multi_experiment_dataset_csv(shuffled_dataset),
        hyperparameters=hyperparameters,
    )

    return x_train, y_train, x_test, y_test


def run(hyperparameters, prepare_dataset, create_model, dryrun=False):
    """ Use the provided hyperparameters to train the model in this module.

    Args:
        hyperparameters: Any variables that are parameterizable for this model. See `get_hyperparameters` for details
        prepare_dataset: A function that takes a raw_dataset and returns (x_train, y_train, x_test, y_test)
        create_model: A function that takes hyperparameters and x_train and returns a compiled model
        dryrun: Whether the model should be run with a tiny dataset in dryrun mode.
    """

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]

    x_train, y_train, x_test, y_test = _get_prepared_dataset(
        prepare_dataset, hyperparameters, dryrun
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

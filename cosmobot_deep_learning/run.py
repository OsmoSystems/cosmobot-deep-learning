import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.load_dataset import load_multi_experiment_dataset_csv

from cosmobot_deep_learning.custom_metrics import (
    magical_incantation_to_make_custom_metric_work,
)


def _initialize_wandb(hyperparameters, y_train, y_test):
    # TODO: do something better? I want metrics defined in hyperparameters, but
    # something chokes when I try to send them to W&B
    hyperparameters_minus_metrics = {
        key: value for key, value in hyperparameters.items() if key != "metrics"
    }

    wandb.init(
        entity="osmo",
        project="cosmobot-do-measurement",
        config={
            "train_sample_count": y_train.shape[0],
            "test_sample_count": y_test.shape[0],
            **hyperparameters_minus_metrics,
        },
    )


def run(hyperparameters, prepare_dataset, create_model):
    """ Use the provided hyperparameters to train the model in this module.

    Args:
        hyperparameters: Any variables that are parameterizable for this model
            epochs: Number of epochs to train for
            batch_size: Training batch size
            model_name: A string label for the model
            dataset_filepath: Filepath (within this package) of the dataset to use for training
            optimizer: Which optimizer function to use
            loss: Which loss function to use
        prepare_dataset: A function that a raw_dataset and returns (x_train, y_train, x_test, y_test)
        create_model: A function that takes hyperparameters and x_train

    """

    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]
    dataset_filepath = hyperparameters["dataset_filepath"]

    x_train, y_train, x_test, y_test = prepare_dataset(
        raw_dataset=load_multi_experiment_dataset_csv(dataset_filepath),
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

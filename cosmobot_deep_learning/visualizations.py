import numpy as np
import pandas as pd
import plotly.graph_objs as go
import wandb


_TRAINING_LOSS_OVER_EPOCHS_TITLE = "Training and Dev loss over epochs"
_DO_PREDICTION_ERROR_TITLE = "DO prediction error"
_ACTUAL_VS_PREDICTED_TITLE = "Actual vs predicted DO"


def _get_loss_over_epochs_figure(training_history):
    """
    Generates a figure that shows loss over epochs for both the dev and training data,
    with their rolling median for the last 200 epochs.
    """
    history_dict = training_history.history

    loss_over_epochs = history_dict["loss"]
    val_loss_over_epochs = history_dict["val_loss"]
    epochs = list(range(1, len(loss_over_epochs) + 1))
    rolling_median_n_epochs = 200

    return go.Figure(
        [
            {"x": epochs, "y": list(val_loss_over_epochs), "name": "Dev loss"},
            {"x": epochs, "y": list(loss_over_epochs), "name": "Training loss"},
            {
                "x": epochs,
                "y": list(
                    pd.Series(val_loss_over_epochs)
                    .rolling(rolling_median_n_epochs)
                    .median()
                ),
                "name": "Dev loss trailing median",
            },
            {
                "x": epochs,
                "y": list(
                    pd.Series(loss_over_epochs)
                    .rolling(rolling_median_n_epochs)
                    .median()
                ),
                "name": "Training loss trailing median",
            },
        ],
        layout={
            "title": _TRAINING_LOSS_OVER_EPOCHS_TITLE,
            "xaxis": {"title": "Epoch"},
            "yaxis": {"title": "Loss (log scale)", "type": "log"},
        },
    )


def _get_actual_vs_predicted_do_figure(
    train_labels, train_predictions, dev_labels, dev_predictions
):
    # NOTE x and y values need to be lists to log correctly
    return go.Figure(
        [
            dict(
                x=list(train_labels),
                y=list(train_predictions),
                mode="markers",
                name="training set",
            ),
            dict(
                x=list(dev_labels),
                y=list(dev_predictions),
                mode="markers",
                name="dev set",
            ),
            dict(x=[0, 160], y=[0, 160], mode="lines", name="truth"),
        ],
        layout={
            "title": _ACTUAL_VS_PREDICTED_TITLE,
            "xaxis": {"title": "YSI DO (mmHg)"},
            "yaxis": {"title": "predicted DO (mmHg)"},
        },
    )


def _get_do_prediction_error_figure(
    train_labels, train_predictions, dev_labels, dev_predictions
):
    # NOTE x and y values need to be lists to log correctly
    return go.Figure(
        [
            dict(
                x=list(train_labels),
                y=list(train_predictions - train_labels),
                mode="markers",
                name="training set",
            ),
            dict(
                x=list(dev_labels),
                y=list(dev_predictions - dev_labels),
                mode="markers",
                name="dev set",
            ),
        ],
        layout={
            "title": _DO_PREDICTION_ERROR_TITLE,
            "xaxis": {"title": "YSI DO (mmHg)"},
            "yaxis": {"title": "error (predicted DO - YSI DO (mmHg))"},
        },
    )


def _log_figure_to_wandb(name, figure):
    wandb.log({name: figure})


def log_loss_over_epochs(history):
    """
    Logs a visualization of "Training loss over epochs" to W&B

    Args:
        history: History object from model.fit()
    """
    _log_figure_to_wandb(
        _TRAINING_LOSS_OVER_EPOCHS_TITLE, _get_loss_over_epochs_figure(history)
    )


def log_do_prediction_error(
    train_labels: np.ndarray,
    train_predictions: np.ndarray,
    dev_labels: np.ndarray,
    dev_predictions: np.ndarray,
):
    """
    Logs a visualization of "DO prediction error" to W&B

    Args:
        train_labels: pandas array of the labels (YSI DO mmHg) for training data
        train_predictions: pandas array of predictions on training data
        dev_labels: pandas array of the labels (YSI DO mmHg) for dev data
        dev_predictions: pandas array of the predictions on dev data
    """
    _log_figure_to_wandb(
        _DO_PREDICTION_ERROR_TITLE,
        _get_do_prediction_error_figure(
            train_labels, train_predictions, dev_labels, dev_predictions
        ),
    )


def log_actual_vs_predicted_do(
    train_labels: np.ndarray,
    train_predictions: np.ndarray,
    dev_labels: np.ndarray,
    dev_predictions: np.ndarray,
):
    """
    Logs a visualization of "Actual vs predicted DO" to W&B

    Args:
        train_labels: pandas array of the labels (YSI DO mmHg) for training data
        train_predictions: pandas array of predictions on training data
        dev_labels: pandas array of the labels (YSI DO mmHg) for dev data
        dev_predictions: pandas array of the predictions on dev data
    """
    _log_figure_to_wandb(
        _ACTUAL_VS_PREDICTED_TITLE,
        _get_actual_vs_predicted_do_figure(
            train_labels, train_predictions, dev_labels, dev_predictions
        ),
    )

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import wandb


_TRAINING_LOSS_OVER_EPOCHS_TITLE = "Training loss over epochs"
_DO_PREDICTION_ERROR_TITLE = "DO prediction error"
_ACTUAL_VS_PREDICTED_TITLE = "Actual vs predicted DO"


def _get_training_loss_over_epochs_figure(training_history):
    history_dict = training_history.history

    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = list(range(1, len(loss_values) + 1))
    rolling_median_n_epochs = 200

    return go.Figure(
        [
            {"x": epochs, "y": list(val_loss_values), "name": "Dev loss"},
            {"x": epochs, "y": list(loss_values), "name": "Training loss"},
            {
                "x": epochs,
                "y": list(
                    pd.Series(val_loss_values).rolling(rolling_median_n_epochs).median()
                ),
                "name": "Dev loss trailing median",
            },
            {
                "x": epochs,
                "y": list(
                    pd.Series(loss_values).rolling(rolling_median_n_epochs).median()
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
                name="Training set",
            ),
            dict(
                x=list(dev_labels),
                y=list(dev_predictions),
                mode="markers",
                name="Dev set",
            ),
            dict(x=[0, 160], y=[0, 160], mode="lines", name="truth"),
        ],
        layout={
            "title": _ACTUAL_VS_PREDICTED_TITLE,
            "xaxis": {"title": "YSI DO (mmHg)"},
            "yaxis": {"title": "Predicted DO (mmHg)"},
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
                name="Training set",
            ),
            dict(
                x=list(dev_labels),
                y=list(dev_predictions - dev_labels),
                mode="markers",
                name="Dev set",
            ),
        ],
        layout={
            "title": _DO_PREDICTION_ERROR_TITLE,
            "xaxis": {"title": "YSI DO (mmHg)"},
            "yaxis": {"title": "Error (Predicted DO - YSI DO (mmHg))"},
        },
    )


def _log_figure_to_wandb(name, figure):
    # TODO is there anything we can do to log to a consistent step value (or not associate these with a step) across runs
    wandb.log({name: figure})


def log_training_loss_over_epochs(history):
    """
    Logs a visualization of "Training loss over epochs" to W&B

    Args:
        history: History object from model.fit()
    """
    _log_figure_to_wandb(
        _TRAINING_LOSS_OVER_EPOCHS_TITLE, _get_training_loss_over_epochs_figure(history)
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

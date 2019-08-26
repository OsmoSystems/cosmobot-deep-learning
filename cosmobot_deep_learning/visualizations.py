import numpy as np
import pandas as pd
import plotly.graph_objs as go
import wandb


_TRAINING_LOSS_OVER_EPOCHS_TITLE = "Training and Dev loss over epochs"
_DO_PREDICTION_ERROR_TITLE = "DO prediction error"
_ACTUAL_VS_PREDICTED_TITLE = "Actual vs predicted DO"

# maximum number of points to allow on a plotly figure that we send to W&B
# turn this down if the plotly charts on W&B are unresponsive
_PLOTLY_WANDB_MAX_POINTS = 4000


def _downsample(array: np.ndarray, max_samples: int):
    """Returns max_samples items from the given array, spread out evenly across indexes.
    """
    if len(array) <= max_samples:
        return array

    downsampled_indexes = np.round(np.linspace(0, len(array) - 1, max_samples)).astype(
        int
    )
    return array[downsampled_indexes]


def _get_loss_over_epochs_figure(training_history):
    """
    Generates a figure that shows loss over epochs for both the dev and training data,
    with their rolling median for the last 200 epochs.
    """
    history_dict = training_history.history

    loss_over_epochs = np.array(history_dict["loss"])
    val_loss_over_epochs = np.array(history_dict["val_loss"])
    epochs = np.array(list(range(1, len(loss_over_epochs) + 1)))
    rolling_median_n_epochs = 200

    val_loss_over_epochs_rolling_median = (
        pd.Series(val_loss_over_epochs).rolling(rolling_median_n_epochs).median()
    )
    loss_over_epochs_rolling_median = (
        pd.Series(loss_over_epochs).rolling(rolling_median_n_epochs).median()
    )

    # downsample for plotly performance
    num_traces = 4  # number of traces on this figure
    max_samples_per_trace = int(_PLOTLY_WANDB_MAX_POINTS / num_traces)
    downsampled_val_loss_over_epochs = _downsample(
        val_loss_over_epochs, max_samples_per_trace
    )
    downsampled_loss_over_epochs = _downsample(loss_over_epochs, max_samples_per_trace)
    downsampled_val_loss_over_epochs_rolling_median = _downsample(
        val_loss_over_epochs_rolling_median, max_samples_per_trace
    )
    downsampled_loss_over_epochs_rolling_median = _downsample(
        loss_over_epochs_rolling_median, max_samples_per_trace
    )
    downsampled_epochs = list(_downsample(epochs, max_samples_per_trace))

    return go.Figure(
        [
            {
                "x": downsampled_epochs,
                "y": list(downsampled_val_loss_over_epochs),
                "name": "Dev loss",
            },
            {
                "x": downsampled_epochs,
                "y": list(downsampled_loss_over_epochs),
                "name": "Training loss",
            },
            {
                "x": downsampled_epochs,
                "y": list(downsampled_val_loss_over_epochs_rolling_median),
                "name": "Dev loss trailing median",
            },
            {
                "x": downsampled_epochs,
                "y": list(downsampled_loss_over_epochs_rolling_median),
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
    num_traces = 2  # traces in this figure
    max_samples_per_trace = int(_PLOTLY_WANDB_MAX_POINTS / num_traces)

    # NOTE x and y values need to be lists to log correctly
    return go.Figure(
        [
            dict(
                x=list(_downsample(train_labels, max_samples_per_trace)),
                y=list(_downsample(train_predictions, max_samples_per_trace)),
                mode="markers",
                name="training set",
            ),
            dict(
                x=list(_downsample(dev_labels, max_samples_per_trace)),
                y=list(_downsample(dev_predictions, max_samples_per_trace)),
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
    num_traces = 2
    max_samples_per_trace = int(_PLOTLY_WANDB_MAX_POINTS / num_traces)

    # NOTE x and y values need to be lists to log correctly
    return go.Figure(
        [
            dict(
                x=list(_downsample(train_labels, max_samples_per_trace)),
                y=list(
                    _downsample(train_predictions - train_labels, max_samples_per_trace)
                ),
                mode="markers",
                name="training set",
            ),
            dict(
                x=list(_downsample(dev_labels, max_samples_per_trace)),
                y=list(
                    _downsample(dev_predictions - dev_labels, max_samples_per_trace)
                ),
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
    figure = _get_loss_over_epochs_figure(history)
    _log_figure_to_wandb(_TRAINING_LOSS_OVER_EPOCHS_TITLE, figure)


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
    figure = _get_do_prediction_error_figure(
        train_labels, train_predictions, dev_labels, dev_predictions
    )
    _log_figure_to_wandb(_DO_PREDICTION_ERROR_TITLE, figure)


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
    figure = _get_actual_vs_predicted_do_figure(
        train_labels, train_predictions, dev_labels, dev_predictions
    )
    _log_figure_to_wandb(_ACTUAL_VS_PREDICTED_TITLE, figure)

import numpy as np
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


def _get_actual_vs_predicted_do_figure(
    train_labels,
    train_predictions,
    dev_labels,
    dev_predictions,
    title=_ACTUAL_VS_PREDICTED_TITLE,
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
            "title": title,
            "xaxis": {"title": "YSI DO (mmHg)"},
            "yaxis": {"title": "predicted DO (mmHg)"},
        },
    )


def _get_do_prediction_error_figure(
    train_labels,
    train_predictions,
    dev_labels,
    dev_predictions,
    title=_DO_PREDICTION_ERROR_TITLE,
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
            "title": title,
            "xaxis": {"title": "YSI DO (mmHg)"},
            "yaxis": {"title": "error (predicted DO - YSI DO (mmHg))"},
        },
    )


def _log_figure_to_wandb(name, figure):
    # Set commit to False to prevent wandb from creating an intermediate step
    wandb.log({name: figure}, commit=False)


def log_do_prediction_error(
    train_labels: np.ndarray,
    train_predictions: np.ndarray,
    dev_labels: np.ndarray,
    dev_predictions: np.ndarray,
    chart_title_annotation: str = "",
):
    """
    Logs a visualization of "DO prediction error" to W&B

    Args:
        train_labels: pandas array of the labels (YSI DO mmHg) for training data
        train_predictions: pandas array of predictions on training data
        dev_labels: pandas array of the labels (YSI DO mmHg) for dev data
        dev_predictions: pandas array of the predictions on dev data
    """
    title = f"{_DO_PREDICTION_ERROR_TITLE}{chart_title_annotation}"
    figure = _get_do_prediction_error_figure(
        train_labels, train_predictions, dev_labels, dev_predictions, title
    )
    _log_figure_to_wandb(title, figure)


def log_actual_vs_predicted_do(
    train_labels: np.ndarray,
    train_predictions: np.ndarray,
    dev_labels: np.ndarray,
    dev_predictions: np.ndarray,
    chart_title_annotation: str = "",
):
    """
    Logs a visualization of "Actual vs predicted DO" to W&B

    Args:
        train_labels: pandas array of the labels (YSI DO mmHg) for training data
        train_predictions: pandas array of predictions on training data
        dev_labels: pandas array of the labels (YSI DO mmHg) for dev data
        dev_predictions: pandas array of the predictions on dev data
    """
    title = f"{_ACTUAL_VS_PREDICTED_TITLE}{chart_title_annotation}"
    figure = _get_actual_vs_predicted_do_figure(
        train_labels, train_predictions, dev_labels, dev_predictions, title
    )
    _log_figure_to_wandb(title, figure)

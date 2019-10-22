import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import Callback

from cosmobot_deep_learning.constants import (
    MG_L_PER_MMHG_AT_25_C_1_ATM as MG_L_PER_MMHG,
)
from cosmobot_deep_learning import visualizations

ARBITRARILY_LARGE_MULTIPLIER = 10

logger = logging.getLogger(__name__)


def _function_namify(_float: float) -> str:
    """ Rounds float to 2 digits and replaces "." with "_"
    e.g. 2.1235 -> 2_12
    """
    return str(round(_float, 2)).replace(".", "_")


def _get_fraction_outside_error_threshold_fn_name(error_threshold_mg_l):
    """ Generates an appropriate function name for the "fraction outside error threshold" custom metric,
    given the desired `error_threshold_mg_l`

    Args:
        error_threshold_mg_l: The error threshold, in mg/L
    """
    return f"fraction_outside_{_function_namify(error_threshold_mg_l)}_mg_l_error"


# Normally I would use functools.partial for this, but keras needs the __name__ attribute, which partials don't have
def get_fraction_outside_error_threshold_fn(
    error_threshold_mg_l, label_scale_factor_mmhg
):
    """ Returns a function that can be used as a keras metric, populated with the appropriate error threshold
    """

    # Ensure that our custom metric uses the same normalizing factor we use to scale our labels
    error_threshold_mmhg = error_threshold_mg_l * MG_L_PER_MMHG
    error_threshold_normalized = error_threshold_mmhg / label_scale_factor_mmhg

    def fraction_outside_error_threshold(y_true, y_pred):
        """ Our custom "satisficing" metric that evaluates what fraction of predictions
            are outside of our acceptable error threshold.

            Aided by:
            https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
        """

        y_pred_error = tf.abs(y_pred - y_true)
        is_outside_error_threshold = tf.greater(
            y_pred_error, error_threshold_normalized
        )

        # count_nonzero counts Trues as not zero and Falses as zero
        count_outside_error_threshold = tf.math.count_nonzero(
            is_outside_error_threshold
        )
        count_total = tf.size(y_true)

        # Cast ints to floats to ensure dtypes match
        fraction_outside = tf.math.divide(
            tf.cast(count_outside_error_threshold, tf.float32),
            tf.cast(count_total, tf.float32),
        )
        return fraction_outside

    fn_name = _get_fraction_outside_error_threshold_fn_name(error_threshold_mg_l)
    fraction_outside_error_threshold.__name__ = fn_name

    return fraction_outside_error_threshold


class ThresholdValMeanAbsoluteErrorOnCustomMetric(Callback):
    """ Keras model callback to add two new metrics
        "val_satisficing_mean_absolute_error" is a filtered version of val_mean_absolute_error,
            only reported when our satisficing metric is hit.
        "val_adjusted_mean_absolute_error" is a modified version of val_mean_absolute_error,
            multiplied by an ARBITRARILY_LARGE_MULTIPLIER when the satisficing metric is not hit
            so that we can evaluate a "best" performing model, prefering the satisficing metric, and
            falling back to the best mean absolute error if the satisficing metric is never reached.
    """

    def __init__(self, acceptable_fraction_outside_error, acceptable_error_mg_l):
        self.acceptable_fraction_outside_error = acceptable_fraction_outside_error

        # The "fraction outside acceptable error" metric depends on what the defined "acceptable error" is
        fraction_outside_acceptable_error_metric = _get_fraction_outside_error_threshold_fn_name(
            acceptable_error_mg_l
        )

        # Use the "val" version of the metric
        self.val_fraction_outside_acceptable_error_metric = (
            f"val_{fraction_outside_acceptable_error_metric}"
        )

    def on_epoch_end(self, epoch, logs=None):
        # Metrics are compiled and average over batches for an entire epoch by Keras
        # in the built-in BaseLogger callback and stored by mutating `logs`, passed on to each subsequent callback.
        # Here we access those epoch-level average metrics and check if the epoch as a whole hit the satisficing metric.
        # Mutate the logs object here as well with this new metric to be picked up in the WandbCallback
        if logs is not None:
            if (
                logs[self.val_fraction_outside_acceptable_error_metric]
                < self.acceptable_fraction_outside_error
            ):
                logs["val_satisficing_mean_absolute_error"] = logs[
                    "val_mean_absolute_error"
                ]
                logs["val_adjusted_mean_absolute_error"] = logs[
                    "val_mean_absolute_error"
                ]
            else:
                logs["val_satisficing_mean_absolute_error"] = None
                logs["val_adjusted_mean_absolute_error"] = (
                    logs["val_mean_absolute_error"] * ARBITRARILY_LARGE_MULTIPLIER
                )


class SaveBestMetricValueAndEpochToWandb(Callback):
    """ Save the best seen value of a particular metric to the Weights & Biases summary for a training run
    This metric is saved as "best_[original metric name]".
    Also stores "best_epoch_by_[original metric name]" with the epoch number that the best metric came from.
    """

    def __init__(self, metric):
        self.source_metric_key = metric
        self.best_metric_key = f"best_{metric}"
        self.best_epoch_key = f"best_epoch_by_{metric}"

    def on_epoch_end(self, epoch, logs):
        current_metric = logs[self.source_metric_key]
        previous_best_metric = wandb.run.summary.get(self.best_metric_key)

        if not previous_best_metric or current_metric < previous_best_metric:
            wandb.run.summary[self.best_metric_key] = current_metric
            wandb.run.summary[self.best_epoch_key] = epoch


class LogPredictionsAndWeights(Callback):
    """ A callback to keep track of best model weights and make predictions to log
        plotly charts to W&B.
        - Holds on to the model weights from the best epoch according to the given metric
          and uses them to predict when the training performance hits a new global minimum
          at the end of an improvement streak.
        - Also makes predictions with the current model every `epoch_interval` epochs.
        - Saves all predictions as `predictions.csv` in the wandb log directory.
        - Saves the final model weights to file (`model-final.h5`) before restoring.

    This assumes that lower is better for the given metric.

    Constructor Args:
        metric: name of the metric to read from logs dict
        dataset: x_train, y_train, x_dev, y_dev dataset tuple (should match the dataset being trained on)
        label_scale_factor_mmhg: scale factor to use when converting predictions and labels for plotly charts
        epoch_interval: Fixed epoch interval to log predictions (default 10)

    Attributes:
        metric: name of the metric to read from logs dict
        best_epoch: epoch number (int) of the best epoch seen so far
        best_value: best value for the given metric seen so far
        best_weights: model weights from the epoch with the best value
        epoch_interval: fixed epoch interval to log predictions
        latest_predictions: cached model predictions to help prevent redundant evaluation
        predictions_file_path: file path where predictions are saved
        label_scale_factor_mmhg: scale factor to use when converting predictions and labels for plotly charts
        x_train, y_train, x_dev, y_dev: dataset slices to make predictions over
    """

    def __init__(
        self,
        metric: str,
        dataset: Tuple,
        label_scale_factor_mmhg: int,
        epoch_interval: int = 10,
    ):
        self.metric = metric
        self.best_epoch = 0
        self.best_value = None
        self.best_weights = None
        self.epoch_interval = epoch_interval

        self.latest_predictions = None
        self.label_scale_factor_mmhg = label_scale_factor_mmhg

        # Save the training and validation datasets to use to generate predictions
        self.x_train, self.y_train, self.x_dev, self.y_dev = dataset

    def _get_predictions(self, weights, x_true, y_true, epoch, training):
        # Prevent memory leak: https://github.com/keras-team/keras/issues/13118
        eval_model = tf.keras.models.clone_model(self.model)
        eval_model.set_weights(weights)

        predictions = eval_model.predict(x_true)

        prediction_count = len(predictions)

        return pd.DataFrame(
            {
                "epoch": [epoch] * prediction_count,
                "current best epoch": [epoch == self.best_epoch] * prediction_count,
                "true DO (mmHg)": y_true.flatten() * self.label_scale_factor_mmhg,
                "predicted DO (mmHg)": predictions.flatten()
                * self.label_scale_factor_mmhg,
                "absolute error (mmHg)": np.abs(predictions - y_true).flatten()
                * self.label_scale_factor_mmhg,
                "training": [training] * prediction_count,
            }
        )

    def _log_predictions(self, weights, epoch):
        training_dataframe = self._get_predictions(
            weights, self.x_train, self.y_train, epoch, training=True
        )
        dev_dataframe = self._get_predictions(
            weights, self.x_dev, self.y_dev, epoch, training=False
        )

        predictions = pd.concat([training_dataframe, dev_dataframe])

        predictions_file_path = os.path.join(wandb.run.dir, "predictions.csv")

        with open(predictions_file_path, "a") as csv_file:
            is_file_empty = csv_file.tell() == 0
            predictions.to_csv(csv_file, index=False, header=is_file_empty, mode="a")

        return predictions

    def _log_predictions_chart(self, predictions, epoch, chart_title_annotation):
        train_labels = predictions[predictions["training"]]["true DO (mmHg)"]
        train_predictions = predictions[predictions["training"]]["predicted DO (mmHg)"]

        dev_labels = predictions[~predictions["training"]]["true DO (mmHg)"]
        dev_predictions = predictions[~predictions["training"]]["predicted DO (mmHg)"]

        visualizations.log_do_prediction_error(
            train_labels,
            train_predictions,
            dev_labels,
            dev_predictions,
            chart_title_annotation,
        )
        visualizations.log_actual_vs_predicted_do(
            train_labels,
            train_predictions,
            dev_labels,
            dev_predictions,
            chart_title_annotation,
        )

    def _rename_model_best(self, epoch: int):
        os.rename(
            os.path.join(wandb.run.dir, "model-best.h5"),
            os.path.join(wandb.run.dir, f"model-best-{epoch}.h5"),
        )

    def _save_final_weights(self):
        # Save final model weights to allow continued training
        self.model.save(os.path.join(wandb.run.dir, "model-final.h5"))

    def _is_periodic_logging_epoch(self, epoch):
        return epoch % self.epoch_interval == 0

    def on_epoch_end(self, epoch, logs):
        current_value = logs[self.metric]

        if self.best_value is None or current_value < self.best_value:
            logger.info(f"New best epoch is {epoch}: ({self.metric} = {current_value})")
            self.best_epoch = epoch
            self.best_value = current_value
            self.best_weights = self.model.get_weights()

        # Model checkpointing and prediction heuristic
        if self.best_epoch == epoch - 1 and self.best_epoch > 0:
            # The previous epoch was the most performant so far
            # We want to check after the current epoch has run so as to only
            # checkpoint/ predict when performance has hit the
            # end of a global improvement streak
            self._rename_model_best(epoch - 1)

            # Use best_weights to log latest predictions (unless the last
            # best_epoch already fell on the regular prediction interval)
            if not self._is_periodic_logging_epoch(self.best_epoch):
                self.latest_predictions = self._log_predictions(
                    self.best_weights, self.best_epoch
                )

            self._log_predictions_chart(
                self.latest_predictions, epoch, chart_title_annotation=" - Best"
            )

        elif self._is_periodic_logging_epoch(epoch):
            # When not at a global minimum, predict on some fixed interval
            self.latest_predictions = self._log_predictions(
                self.model.get_weights(), epoch
            )

    def on_train_end(self, logs=None):
        self._save_final_weights()

        # epochs in params is the total number of epochs, but the epochs themselves are 0-indexed
        final_epoch = self.params["epochs"] - 1

        if not self._is_periodic_logging_epoch(final_epoch):
            # Make final predictions and upload one last set of charts
            self.latest_predictions = self._log_predictions(
                self.model.get_weights(), final_epoch
            )

        self._log_predictions_chart(
            self.latest_predictions, final_epoch, chart_title_annotation=" - Final"
        )

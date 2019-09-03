import keras
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf

from cosmobot_deep_learning.constants import (
    MG_L_PER_MMHG_AT_25_C_1_ATM as MG_L_PER_MMHG,
)

ARBITRARILY_LARGE_MULTIPLIER = 10


def _function_namify(_float):
    """ Rounds float to 2 digits and replaces "." with "_"
    """
    return str(round(_float, 2)).replace(".", "_")


# Normally I would use functools.partial for this, but keras needs the __name__ attribute, which partials don't have
def get_fraction_outside_acceptable_error_fn(
    acceptable_error_mg_l, label_scale_factor_mmhg
):
    """ Returns a function that can be used as a keras metric, populated with the appropriate acceptable_error threshold
    """

    # Ensure that our custom metric uses the same normalizing factor we use to scale our labels
    acceptable_error_mmhg = acceptable_error_mg_l * MG_L_PER_MMHG
    acceptable_error_normalized = acceptable_error_mmhg / label_scale_factor_mmhg

    def fraction_outside_acceptable_error(y_true, y_pred):
        """ Our custom "satisficing" metric that evaluates what fraction of predictions
            are outside of our acceptable error threshold.

            Aided by:
            https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
        """

        y_pred_error = tf.abs(y_pred - y_true)
        is_outside_acceptable_error = tf.greater(
            y_pred_error, acceptable_error_normalized
        )

        # count_nonzero counts Trues as not zero and Falses as zero
        count_outside_acceptable_error = tf.count_nonzero(is_outside_acceptable_error)
        count_total = tf.size(y_true)

        # Cast to float so that the division calculation returns a float (tf uses Python 2 semantics)
        fraction_outside = tf.div(
            tf.cast(count_outside_acceptable_error, tf.float32),
            tf.cast(count_total, tf.float32),
        )
        return fraction_outside

    fraction_outside_acceptable_error.__name__ = (
        f"fraction_outside_{_function_namify(acceptable_error_mg_l)}_mg_l_error"
    )

    return fraction_outside_acceptable_error


class MmhgErrorAtPercentile(Callback):
    """ Keras model callback to add two new metrics
        "error_at_percentile_mmhg" is ... TODO
        "val_error_at_percentile_mmhg" is ... TODO

    Args:
        percentile: Calculate the error threshold that percentile of predictions fall within
        label_scale_factor_mmhg: The scaling factor to use to scale the returned error threshold by
            (to reverse the normalization effect)
        dataset: a tuple of (x_train, y_train, x_val, y_val)
        epoch_interval: Only calculate this metric once every epoch_interval
    """

    def __init__(self, percentile, label_scale_factor_mmhg, dataset, epoch_interval=10):
        super(Callback, self).__init__()

        self.percentile = percentile
        self.label_scale_factor_mmhg = label_scale_factor_mmhg
        self.epoch_interval = epoch_interval

        # Save the training and validation datasets to use to generate predictions
        self.x_train, self.y_train, self.x_val, self.y_val = dataset

    def error_at_percentile_mmhg(self, x_true, y_true):
        """ Calculate the error (in mmhg) that self.percentile of predictions fall within.
        """
        y_pred = self.model.predict(x_true)
        y_pred_error = np.abs(y_pred - y_true)

        normalized_error_at_percentile = np.percentile(y_pred_error, q=self.percentile)
        return normalized_error_at_percentile * self.label_scale_factor_mmhg

    def on_epoch_end(self, epoch, logs=None):
        # Metrics are compiled and averaged over batches for an entire epoch by Keras
        # in the built-in BaseLogger callback and stored by mutating `logs`, passed on to each subsequent callback.
        # To add our own custom metric that is computed per-epoch instead, calculate it here and add it to logs

        # Only calculate once per interval of epochs, since the prediction is expensive
        skip_epoch = epoch % self.epoch_interval

        if logs is None or skip_epoch:
            return

        error_at_percentile_mmhg = self.error_at_percentile_mmhg(
            x_true=self.x_train, y_true=self.y_train
        )
        val_error_at_percentile_mmhg = self.error_at_percentile_mmhg(
            x_true=self.x_val, y_true=self.y_val
        )

        p = int(self.percentile)

        # fmt: off
        logs[f"error_at_{p}_percentile_mmhg"] = error_at_percentile_mmhg
        logs[f"val_error_at_{p}_percentile_mmhg"] = val_error_at_percentile_mmhg
        logs[f"error_at_{p}_percentile_mg_l"] = error_at_percentile_mmhg / MG_L_PER_MMHG
        logs[f"val_error_at_{p}_percentile_mg_l"] = val_error_at_percentile_mmhg / MG_L_PER_MMHG
        # fmt: on


def magical_incantation_to_make_custom_metric_work():
    """ This magical incantation must be called before model.fit() to make our custom metric work
        I honestly have no idea why this makes our custom metric work... but it does.
        https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    """
    keras.backend.get_session().run(tf.local_variables_initializer())


class ThresholdValMeanAbsoluteErrorOnCustomMetric(Callback):
    """ Keras model callback to add two new metrics
        "val_satisficing_mean_absolute_error" is a filtered version of val_mean_absolute_error,
            only reported when our satisficing metric is hit.
        "val_adjusted_mean_absolute_error" is a modified version of val_mean_absolute_error,
            multiplied by an ARBITRARILY_LARGE_MULTIPLIER when the satisficing metric is not hit
            so that we can evaluate a "best" performing model, prefering the satisficing metric, and
            falling back to the best mean absolute error if the satisficing metric is never reached.

    """

    def __init__(self, acceptable_fraction_outside_error):
        self.acceptable_fraction_outside_error = acceptable_fraction_outside_error

    def on_epoch_end(self, epoch, logs=None):
        # Metrics are compiled and average over batches for an entire epoch by Keras
        # in the built-in BaseLogger callback and stored by mutating `logs`, passed on to each subsequent callback.
        # Here we access those epoch-level average metrics and check if the epoch as a whole hit the satisficing metric.
        # Mutate the logs object here as well with this new metric to be picked up in the WandbCallback
        if logs is not None:
            if (
                # TODO: fix this hardcoding
                logs["val_fraction_outside_0_5_mg_l_error"]
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

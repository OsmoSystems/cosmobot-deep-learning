import keras
from keras.callbacks import Callback
import tensorflow as tf


# Normally I would use functools.partial for this, but keras needs the __name__ attribute, which partials don't have
def get_fraction_outside_acceptable_error_fn(acceptable_error):
    """ Returns a function that can be used as a keras metric, populated with the appropriate threshold
    """

    def fraction_outside_acceptable_error(y_true, y_pred):
        """ Our custom "satisficing" metric that evaluates what fraction of predictions
            are outside of our acceptable error threshold.

            Aided by:
            https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
        """

        y_pred_error = tf.abs(y_pred - y_true)
        is_outside_acceptable_error = tf.greater(y_pred_error, acceptable_error)

        # count_nonzero counts Trues as not zero and Falses as zero
        count_outside_acceptable_error = tf.count_nonzero(is_outside_acceptable_error)
        count_total = tf.size(y_true)

        # Cast to float so that the division calculation returns a float (tf uses Python 2 semantics)
        fraction_outside = tf.div(
            tf.cast(count_outside_acceptable_error, tf.float32),
            tf.cast(count_total, tf.float32),
        )
        return fraction_outside

    return fraction_outside_acceptable_error


def magical_incantation_to_make_custom_metric_work():
    """ This magical incantation must be called before model.fit() to make our custom metric work
        I honestly have no idea why this makes our custom metric work... but it does.
        https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    """
    keras.backend.get_session().run(tf.local_variables_initializer())


class FilterCustomMetricCallback(Callback):
    """ Keras model callback to add a new metric which is a filtered version of mean_absolute_error,
        only reported when our satisficing metric is hit.

        For epoch when the satisficing metric is not hit, sets this custom metric to Inf.
    """

    custom_metric_mappings = [
        (
            "fraction_outside_acceptable_error",
            "satisficing_mean_absolute_error",
            "mean_absolute_error",
        ),
        (
            "val_fraction_outside_acceptable_error",
            "val_satisficing_mean_absolute_error",
            "val_mean_absolute_error",
        ),
    ]

    def __init__(self, acceptable_error_fraction):
        self.acceptable_error_fraction = acceptable_error_fraction

    def on_epoch_end(self, epoch, logs=None):
        # Metrics are compiled and average over batchs for an entire epoch by Keras
        # in the built-in BaseLogger callback and stored by mutating `logs`, passed on to each subsequent callback.
        # Here we access those epoch-level average metrics and check if the epoch as a whole hit the satisficing metric.
        # Mutate the logs object here as well with this new metric to be picked up in the WandbCallback
        if logs is not None:
            for (
                metric_to_evaluate,
                metric_to_set,
                metric_to_match,
            ) in self.custom_metric_mappings:
                if set([metric_to_evaluate, metric_to_match]).issubset(
                    set(logs.keys())
                ):
                    if logs[metric_to_evaluate] < self.acceptable_error_fraction:
                        logs[metric_to_set] = logs[metric_to_match]
                    else:
                        logs[metric_to_set] = float("Inf")

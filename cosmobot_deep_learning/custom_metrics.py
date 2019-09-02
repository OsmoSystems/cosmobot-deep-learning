import keras
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


def get_error_at_percentile_fn(acceptable_fraction_outside_error, label_scale_factor):
    """ Returns a function that can be used as a keras metric, populated with the appropriate percentile

    Args:
        acceptable_fraction_outside_error: Set the percentile to calculate the error threshold that
            (1 - acceptable_fraction_outside_error) fraction of predictions fall within
        label_scale_factor: The scaling factor to use to scale the returned error threshold by (to reverse the normalization effect)
    """
    # Massage acceptable_fraction_outside_error into a percentile form, so that ultimately we are determining
    # the error bar that X% of our predictions fall within
    acceptable_fraction_within_error = 1 - acceptable_fraction_outside_error
    percentile = 100 * acceptable_fraction_within_error

    def error_at_percentile(y_true, y_pred):
        """ A custom "satisficing" metric that calculates the error that 95% of our predictions fall within.
        """
        y_pred_error = tf.abs(y_pred - y_true)

        normalized_error_at_percentile = tf.contrib.distributions.percentile(
            y_pred_error, q=percentile
        )

        return normalized_error_at_percentile * label_scale_factor

    return error_at_percentile


def magical_incantation_to_make_custom_metric_work():
    """ This magical incantation must be called before model.fit() to make our custom metric work
        I honestly have no idea why this makes our custom metric work... but it does.
        https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    """
    keras.backend.get_session().run(tf.local_variables_initializer())

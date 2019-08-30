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


def get_satisficing_mean_absolute_error_fn(acceptable_error, acceptable_error_fraction):
    """ Returns a function that can be used as a keras metric, populated with the appropriate threshold
    """

    def satisficing_mean_absolute_error(y_true, y_pred):
        """ Mean absolute error metric "satisficing" metric that evaluates what fraction of predictions
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

        # If the satisficing metric is hit, return the mean absolute error
        if fraction_outside < acceptable_error_fraction:
            return tf.mean(y_pred_error)
        # Otherwise return 1
        return 1

    return satisficing_mean_absolute_error


def magical_incantation_to_make_custom_metric_work():
    """ This magical incantation must be called before model.fit() to make our custom metric work
        I honestly have no idea why this makes our custom metric work... but it does.
        https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    """
    keras.backend.get_session().run(tf.local_variables_initializer())

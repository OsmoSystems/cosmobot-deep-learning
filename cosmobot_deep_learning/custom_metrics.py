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

        count_outside_acceptable_error = tf.reduce_sum(
            # Cast bools to floats to make them countable *shrug*
            tf.cast(is_outside_acceptable_error, tf.float32)
        )

        # Cast to float so that the division calculation returns a float (tf uses Python 2 semantics)
        count_total = tf.cast(tf.size(y_true), tf.float32)

        fraction_outside = tf.div(count_outside_acceptable_error, count_total)
        return fraction_outside

    return fraction_outside_acceptable_error


def magical_incantation_to_make_custom_metric_work():
    """ This magical incantation must be called before model.fit() to make our custom metric work
        I honestly have no idea why this makes our custom metric work... but it does.
        https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    """
    keras.backend.get_session().run(tf.local_variables_initializer())

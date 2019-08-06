import tensorflow as tf


# Normally I would use functools.partial for this, but keras needs the __name__ attribute, which partials don't have
def get_fraction_outside_acceptable_error_fn(acceptable_error):
    """ Returns a function that can be used as a keras metric, populated with the appropriate threshold
    """

    def fraction_outside_acceptable_error(y, y_predicted):
        """ Our custom "satisficing" metric that evaluates what fraction of predictions
            are outside of our acceptable error threshold.

            Aided by:
            https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
        """
        _, fraction_below = tf.compat.v1.metrics.percentage_below(
            tf.abs(y_predicted - y), threshold=acceptable_error
        )

        # We actually care about the fraction above
        return 1 - fraction_below

    return fraction_outside_acceptable_error

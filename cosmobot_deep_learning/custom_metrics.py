import tensorflow as tf


# TODO: check math
# Convert from our mg/l standard to mmHg
_ACCEPTABLE_ERROR_MG_L = 0.5

# In fresh water at 25 degC and 1 ATM, saturated DO is 8.3 mg/L
# (https://www.engineeringtoolbox.com/oxygen-solubility-water-d_841.html)
# Partial pressure of DO at 1 ATM should be 0.2096 ATM = 160 mmHg
# Therefore, at 25 degC and 1 ATM, the conversion from mg/L to mmHg should be:
# 160/8.3 = 19.28 mmHg / (mg/L)
# Therefore, acceptable error in mmHg is: 0.5 (mg/L) * (19.28) = 9.64 mmHg
_ACCEPTABLE_ERROR_MMHG = 9.64

# Using the same normalizing scaling factor, divide by DO_SCALE_FACTOR:
# 9.64 mmHg / 160 mmHg = 0.06
_ACCEPTABLE_ERROR_NORMALIZED = 0.06


# Huh, it looks like the mmHg conversion actually just factors out, and we get
# the same answer by just finding the fraction 0.5 mg/L / 8.3 mg/L


# TODO:
#  - would we rather define this as fraction_within_acceptable_error?
#  - do we want some way to report what the _ACCEPTABLE_ERROR_NORMALIZED was for this run?
#    in case we change it, or later realize the math was wrong?
def fraction_outside_acceptable_error(y, y_predicted):
    """ Our custom "satisficing" metric that evaluates what percentage of predictions
        were outside of our acceptable error threshold.
        
        Aided by:
        https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
    """
    tf_metric = tf.compat.v1.metrics.percentage_below(
        tf.abs(y_predicted - y), threshold=_ACCEPTABLE_ERROR_NORMALIZED
    )

    # We actually care about the percentage above
    return 1 - tf_metric[1]

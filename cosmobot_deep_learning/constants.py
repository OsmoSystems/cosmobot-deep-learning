from tensorflow import keras

# OXYGEN_FRACTION_IN_ATMOSPHERE = 0.2096
# ATMOSPHERIC_PRESSURE_MMHG = 760
# ATMOSPHERIC_OXYGEN_PRESSURE_MMHG = (
#     ATMOSPHERIC_PRESSURE_MMHG * OXYGEN_FRACTION_IN_ATMOSPHERE
# )

# The calculation above would give us 159.296, but in many places we've just been using 160
ATMOSPHERIC_OXYGEN_PRESSURE_MMHG = 160

# In fresh water at 25 degC and 1 ATM, saturated DO is 8.3 mg/L
# (https://www.engineeringtoolbox.com/oxygen-solubility-water-d_841.html)
DO_CONCENTRATION_25_C_1_ATM_MG_L = 8.3
MG_L_PER_MMHG_AT_25_C_1_ATM = (
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG / DO_CONCENTRATION_25_C_1_ATM_MG_L
)

# Our current definition of acceptable error is that 95% of our predictions are within 0.5 mg/L
ACCEPTABLE_FRACTION_OUTSIDE_ERROR = 0.05
ACCEPTABLE_ERROR_MG_L = 0.5

# Protocol 4 supports large datasets, unlike 3 which is the default
LARGE_FILE_PICKLE_PROTOCOL = 4


OPTIMIZER_CLASSES_BY_NAME = {
    "adam": keras.optimizers.Adam,
    "adadelta": keras.optimizers.Adadelta,
}

ACTIVATION_LAYER_BY_NAME = {
    "relu": keras.layers.ReLU,
    "leakyrelu": keras.layers.LeakyReLU,
    "prelu": keras.layers.PReLU,
    "linear": lambda: keras.layers.Activation(activation="linear"),
    "sigmoid": lambda: keras.layers.Activation(activation="sigmoid"),
}

AUTO_ASSIGN_GPU = "auto"

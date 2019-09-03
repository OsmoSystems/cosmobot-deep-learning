from enum import Enum


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
MG_L_TO_MMHG_AT_25_C_1_ATM = (
    ATMOSPHERIC_OXYGEN_PRESSURE_MMHG / DO_CONCENTRATION_25_C_1_ATM_MG_L
)

# Our current definition of acceptable error is that 95% of our predictions are within 0.5 mg/L
ACCEPTABLE_ERROR_MG_L = 0.5


class OptimizerName(Enum):
    ADAM = "Adam"
    ADADELTA = "AdaDelta"

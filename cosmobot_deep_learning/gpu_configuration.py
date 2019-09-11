import keras
import tensorflow


def dont_use_all_the_gpu_memory():
    """Set up Keras/TensorFlow to allow multiple models to be trained on one GPU"""

    config = tensorflow.ConfigProto()
    # The default is to "not allow growth", which is achieved by reserving all the RAM for a GPU from the outset
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tensorflow.Session(config=config))

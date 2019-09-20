from typing import List, Dict, Union

import keras
import shap
import numpy as np
from sklearn.manifold import TSNE

from cosmobot_deep_learning.hyperparameters import DEFAULT_LABEL_SCALE_FACTOR_MMHG


def _get_custom_objects(metrics: List) -> Dict:
    """ Get custom objects necessary to load a keras Model from .h5 file.
        Currently this is just our custom metric functions.
    """
    custom_metrics = {metric.__name__: metric for metric in metrics if callable(metric)}
    return custom_metrics


def load_model_from_h5(run_hyperparameters: Dict, model_filepath: str) -> keras.Model:
    """ Load a .h5 model file as a keras Model

        Args:
            run_hyperparameters: A dictionary containing hyperparameters necessary for deserializing the model,
                currently:
                    "metrics": The list of metric names and custom metric functions compiled into the model.
            model_filepath: The filepath to the .h5 file to load.
        Returns:
            A keras Model
    """
    custom_objects = _get_custom_objects(run_hyperparameters["metrics"])
    return keras.models.load_model(model_filepath, custom_objects=custom_objects)


def get_submodel_for_input(
    model: keras.Model, last_layer: str, input_index: int = None
) -> keras.Model:
    """ Pull out a subsection from an input model into a standalone Model for independent analysis.

        Args:
            model: A keras Model
            last_layer: The name of the final layer to stop at. This will be the output layer of the new model.
            input_index: Optional. The input index of a branch to pull out if providing a multi-branch model.
        Returns:
            A new single-input keras Model up to the specified layer.
    """
    # If there are multiple input branches, get the one specified by input_index
    if type(model.input) == "list":
        input = model.input[input_index]
    else:
        input = model.input

    return keras.Model(inputs=input, outputs=model.get_layer(last_layer).output)


def get_layer_tsne_vectors_for_images(
    model: keras.Model,
    layer_to_visualize: str,
    images: np.ndarray,
    image_input_index: int = 0,
    random_state: int = 0,
) -> np.ndarray:
    """ Compute tSNE values for feature vectors of images generated at a particular layer of a keras Model

        Args:
            model: A keras Model
            layer_to_visualize: The name of the model layer whose output should be visualized
            images: A numpy array of RGB images to be passed into the model
            image_input_index: Integer index of the model's input convolutional branch to be evaluated.
            random_state: Random seed to pass in to tSNE
        Returns:
            2 dimensional numpy array of tSNE similarity scores for each image
    """
    submodel = get_submodel_for_input(model, layer_to_visualize, image_input_index)

    feature_vector = submodel.predict(images)

    return TSNE(random_state=random_state).fit_transform(feature_vector)


def plot_shap_deep_explainer(
    model: keras.Model,
    x_data: Union[np.ndarray, List[np.ndarray]],
    y_data: np.ndarray,
    num_images_to_plot,
    image_input_index: int = 1,
    numerical_input_index: int = 0,
    numerical_input_labels: List = None,
    label_scale_factor_mmhg: int = DEFAULT_LABEL_SCALE_FACTOR_MMHG,
):
    """ Generate SHAP values for a given model and inputs and plot them visually.
        Image input SHAP values are rendered on sample images, numerical input SHAP values are scatter plotted

        Args:
            model: The keras Model to evaluate
            x_data: The x input data to the model. Can either be a numpy array of inputs, or a list of inputs,
                as appropriate for the given model
            y_data: The label values corresponding to the x_data
            num_images_to_plot: The number of input images to plot pixel SHAP values for
            image_input_index: Integer index of the model's input convolutional branch to be evaluated
            numerical_input_index: Integer index of the model's numerical input to be evaluated
            numerical_input_labels: Human-readable names corresponding the the numerical inputs
            label_scale_factor_mmhg: Scale factor to use when displaying human-readable label values
    """

    deep_explainer = shap.DeepExplainer(model, x_data)

    shap_values = deep_explainer.shap_values(x_data)

    format_do_values = np.vectorize(
        lambda target_value: f"{np.round(target_value * label_scale_factor_mmhg)} mmHg"
    )
    labels = format_do_values(y_data[:num_images_to_plot])

    shap.image_plot(
        [shap_values[0][image_input_index][:num_images_to_plot]],
        x_data[image_input_index][:num_images_to_plot],
        labels=labels,
    )

    if numerical_input_index is not None:
        shap.summary_plot(
            shap_values[0][numerical_input_index],
            x_data[numerical_input_index],
            feature_names=numerical_input_labels,
        )

# TODO tests

import argparse
import importlib
import logging
import os
import sys
import tempfile

import pandas as pd
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback

from cosmobot_deep_learning.custom_metrics import (
    get_fraction_outside_error_threshold_fn,
    LogPredictionsAndWeights,
    ThresholdValMeanAbsoluteErrorOnCustomMetric,
)
from cosmobot_deep_learning.load_dataset import (
    download_images_and_attach_filepaths_to_dataset,
    get_pkg_dataset_filepath,
)
from cosmobot_deep_learning.preprocess_image import (
    fix_multiprocessing_with_keras_on_macos,
)


class RunFileDoesNotExist(Exception):
    pass


class FileAlreadyExists(Exception):
    pass


def _download_run_file(run, filename, destination_directory="."):
    try:
        run_file = run.file(filename)
    except wandb.apis.CommError as e:
        if f"{filename} does not exist" in e.message:
            raise RunFileDoesNotExist(
                f"File {filename} does not exist for run {run.id}"
            )
        raise

    full_destination_path = os.path.join(destination_directory, filename)
    if os.path.exists(full_destination_path):
        raise FileAlreadyExists(f"File {full_destination_path} already exists locally.")

    run_file.download(root=destination_directory)


class ModelBestH5File:
    """Context manager that downloads a run's best model h5 file to a temporary location
    and provides the full path to the file.

    Deletes the file on exit.
    """

    best_model_filename = "model-best.h5"

    def __init__(self, run):
        """
        Args:
            run: wandb.Api run
        """
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix="cosmobot_deep_learning_rehydrate_"
        )

        _download_run_file(
            run, self.best_model_filename, destination_directory=self.temp_dir.name
        )

    def __enter__(self):
        return os.path.join(self.temp_dir.name, self.best_model_filename)

    def __exit__(self, type, value, traceback):
        self.temp_dir.cleanup()


def _load_untrainable_model(hyperparameters, model_h5_filepath):
    """Returns a compiled keras model loaded from the given h5 filepath.
    The model will be untrainable (weights will not change with model.fit).
    """

    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    metric_names = hyperparameters["metrics"]

    # mapping of custom metric names to their function for load_model()
    custom_objects = {
        "fraction_outside_0_1_mg_l_error": get_fraction_outside_error_threshold_fn(
            0.1, label_scale_factor_mmhg
        ),
        "fraction_outside_0_3_mg_l_error": get_fraction_outside_error_threshold_fn(
            0.3, label_scale_factor_mmhg
        ),
        "fraction_outside_0_5_mg_l_error": get_fraction_outside_error_threshold_fn(
            0.5, label_scale_factor_mmhg
        ),
    }

    model = keras.models.load_model(
        model_h5_filepath, custom_objects=custom_objects, compile=False
    )

    # freeze layers because we're going to (ab)use fit() and don't want it to learn
    # NOTE: training and validation performance may still differ if there is dropout
    for layer in model.layers:
        layer.trainable = False
    model.trainable = False

    # get the functions for custom metrics
    metrics = [
        custom_objects.get(metric_name, metric_name) for metric_name in metric_names
    ]

    # need to compile _after_ setting trainable = False
    model.compile(loss=hyperparameters["loss"], metrics=metrics)
    return model


def _get_prepared_dataset(
    model_name, hyperparameters, dataset_filename, dataset_sampling_column=None
):
    prepare_dataset_for_model = _get_prepare_dataset_fn_for_model(model_name)

    dataset_path = get_pkg_dataset_filepath(dataset_filename)
    raw_dataset = pd.read_csv(dataset_path)

    if dataset_sampling_column:
        raw_dataset = raw_dataset[raw_dataset[dataset_sampling_column]]

    # TODO remove
    raw_dataset = raw_dataset[0:10]

    downloaded_dataset = download_images_and_attach_filepaths_to_dataset(raw_dataset)
    x, y = prepare_dataset_for_model(downloaded_dataset, hyperparameters)
    return (x, y)


def _get_run(run_id):
    entity = "osmo"
    project = "cosmobot-do-measurement"
    run_path = f"{entity}/{project}/{run_id}"

    api = wandb.Api()
    return api.run(run_path)


def _get_prepare_dataset_fn_for_model(model_name):
    """Returns the PREPARE_DATASET_FUNCTION value from the model module for the
    given module_name.
    """
    model_module = importlib.import_module(
        f"cosmobot_deep_learning.models.{model_name}"
    )
    # TODO nice error if module doesn't have PREPARE_DATASET_FUNCTION defined
    return model_module.PREPARE_DATASET_FUNCTION


def _evaluate_model(run_id, dataset_filename, dataset_sampling_column=None):
    fix_multiprocessing_with_keras_on_macos()

    run = _get_run(run_id)
    hyperparameters = run.config

    # no need to run this on gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with ModelBestH5File(run) as model_h5_filepath:
        model = _load_untrainable_model(hyperparameters, model_h5_filepath)

    # TODO support cached datasets?
    model_name = run.config["model_name"]
    x, y = _get_prepared_dataset(
        model_name, hyperparameters, dataset_filename, dataset_sampling_column
    )

    wandb.init(
        config={
            "run_id": run_id,
            "dataset_filename": dataset_filename,
            "dataset_sampling_column": dataset_sampling_column,
        },
        tags=["model-evaluation"],
    )

    batch_size = hyperparameters["batch_size"]
    label_scale_factor_mmhg = hyperparameters["label_scale_factor_mmhg"]
    acceptable_error_mg_l = hyperparameters["acceptable_error_mg_l"]
    acceptable_fraction_outside_error = hyperparameters[
        "acceptable_fraction_outside_error"
    ]

    # TODO return model and prepared dataset? think about how this could be used in a jupyter notebook
    model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=1,
        verbose=2,
        validation_data=(x, y),
        callbacks=[
            ThresholdValMeanAbsoluteErrorOnCustomMetric(
                acceptable_fraction_outside_error=acceptable_fraction_outside_error,
                acceptable_error_mg_l=acceptable_error_mg_l,
            ),
            WandbCallback(verbose=1, monitor="val_adjusted_mean_absolute_error"),
            LogPredictionsAndWeights(
                metric="val_adjusted_mean_absolute_error",
                dataset=(x, y, x, y),  # TODO is this ok?
                label_scale_factor_mmhg=label_scale_factor_mmhg,
            ),
        ],
    )


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Evaluate model performance.")
    parser.add_argument(
        "wandb_run_id",
        help="W&B id of a run in the osmo/cosmobot-do-measurement project",
    )
    parser.add_argument(
        "dataset_filename",
        help="Name of dataset file in cosmobot_deep_learning/datasets directory",
    )
    parser.add_argument(
        "--sampling-column-name",
        "-c",
        help="Boolean column in the dataset that identifies samples to use.",
    )
    return parser.parse_args(args)


# TODO move this somewhere so that it's shared with run.py?
def _initialize_logging():
    logging_format = "%(asctime)s [%(levelname)s]--- %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=logging_format, handlers=[logging.StreamHandler()]
    )


def main(args):
    _initialize_logging()
    parsed_args = _parse_args(args)
    _evaluate_model(
        parsed_args.wandb_run_id,
        parsed_args.dataset_filename,
        parsed_args.sampling_column_name,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

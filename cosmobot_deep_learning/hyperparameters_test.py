from unittest.mock import sentinel

import pytest

from . import hyperparameters as module

MOCK_HYPERPARAMETERS = {
    "model_name": sentinel.something,
    "dataset_filename": "dataset.csv",
    "epochs": sentinel.something,
    "batch_size": sentinel.something,
    "optimizer": sentinel.something,
    "loss": sentinel.something,
    "input_columns": sentinel.something,
    "label_column": sentinel.something,
    "label_scale_factor_mmhg": 100,
}


@pytest.fixture
def mock_dataset_fns(mocker):
    mocker.patch.object(
        module, "get_pkg_dataset_filepath", return_value=sentinel.dataset_filepath
    )
    mocker.patch.object(module, "get_dataset_hash", return_value=sentinel.dataset_hash)


REQUIRED_HYPERPARAMETERS = [
    "model_name",
    "dataset_filename",
    "input_columns",
    "label_column",
    "label_scale_factor_mmhg",
    "epochs",
    "batch_size",
    "optimizer",
    "loss",
    "acceptable_error_mg_l",
    "acceptable_error_mmhg",
    "dataset_filepath",
    "dataset_hash",
    "acceptable_error_normalized",
]


class TestGetHyperparameters:
    def test_includes_all_required_attributes(self, mock_dataset_fns):
        actual = module.get_hyperparameters(**MOCK_HYPERPARAMETERS)

        missing_attributes = set(REQUIRED_HYPERPARAMETERS).difference(actual.keys())
        assert not missing_attributes

    def test_calculates_dataset_attributes(self, mock_dataset_fns):
        actual = module.get_hyperparameters(**MOCK_HYPERPARAMETERS)
        assert actual["dataset_filepath"] == sentinel.dataset_filepath
        assert actual["dataset_hash"] == sentinel.dataset_hash

    def test_calculates_acceptable_errors(self, mock_dataset_fns):
        actual = module.get_hyperparameters(**MOCK_HYPERPARAMETERS)
        assert actual["acceptable_error_mg_l"] == 0.5
        assert actual["acceptable_error_mmhg"] == 9.638554216867469
        assert actual["acceptable_error_normalized"] == 0.09638554216867469

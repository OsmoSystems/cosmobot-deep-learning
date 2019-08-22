from unittest.mock import sentinel

import pytest

from . import hyperparameters as module

MIN_MODEL_HYPERPARAMETERS = {
    "model_name": sentinel.something,
    "dataset_filename": "dataset.csv",
    "numerical_input_columns": sentinel.something,
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
    "numerical_input_columns",
    "label_column",
    "label_scale_factor_mmhg",
    "epochs",
    "batch_size",
    "optimizer",
    "loss",
    "acceptable_error_mg_l",
    "acceptable_error_mmhg",
    "acceptable_error_normalized",
    "dataset_filepath",
    "dataset_hash",
]


class TestGetHyperparameters:
    def test_includes_all_required_attributes(self, mock_dataset_fns):
        actual = module.get_hyperparameters(**MIN_MODEL_HYPERPARAMETERS)

        missing_attributes = set(REQUIRED_HYPERPARAMETERS).difference(actual.keys())
        assert not missing_attributes


class TestCalculateHyperparameters:
    def test_calculates_dataset_attributes(self, mock_dataset_fns):
        actual = module._calculate_additional_hyperparameters(
            dataset_filename=sentinel.dataset_filename,
            acceptable_error_mg_l=0.5,
            label_scale_factor_mmhg=100,
        )
        assert actual["dataset_filepath"] == sentinel.dataset_filepath
        assert actual["dataset_hash"] == sentinel.dataset_hash

    def test_calculates_acceptable_errors(self, mock_dataset_fns):
        actual = module._calculate_additional_hyperparameters(
            dataset_filename=sentinel.dataset_filename,
            acceptable_error_mg_l=0.5,
            label_scale_factor_mmhg=100,
        )
        assert actual["acceptable_error_mmhg"] == 9.638554216867469
        assert actual["acceptable_error_normalized"] == 0.09638554216867469


class TestGuardNoOverriddenCalculatedHyperparameters:
    def test_raises_if_overridden(self):
        with pytest.raises(ValueError):
            module._guard_no_overridden_calculated_hyperparameters(
                calculated={
                    "calculated": sentinel.calculated,
                    "overridden": sentinel.overridden,
                },
                model_specific={
                    "overridden": sentinel.overridden,
                    "model_specific": sentinel.model_specific,
                },
            )

    def test_doesnt_raise_if_not_overridden(self):
        module._guard_no_overridden_calculated_hyperparameters(
            calculated={
                "calculated": sentinel.calculated,
                "overridden": sentinel.overridden,
            },
            model_specific={"model_specific": sentinel.model_specific},
        )

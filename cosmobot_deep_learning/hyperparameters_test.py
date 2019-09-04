from enum import Enum
from unittest.mock import call, Mock, sentinel

import pytest

from . import hyperparameters as module

MIN_MODEL_HYPERPARAMETERS = {
    "model_name": sentinel.something,
    "dataset_filename": "dataset.csv",
    "numeric_input_columns": sentinel.something,
}


@pytest.fixture
def mock_dataset_fns(mocker):
    mocker.patch.object(
        module, "get_pkg_dataset_filepath", return_value=sentinel.dataset_filepath
    )
    mock_get_dataset_csv_hash = mocker.patch.object(
        module, "get_dataset_csv_hash", return_value=sentinel.dataset_hash
    )
    return {"mock_get_dataset_csv_hash": mock_get_dataset_csv_hash}


REQUIRED_HYPERPARAMETERS = [
    "model_name",
    "dataset_filename",
    "numeric_input_columns",
    "label_column",
    "label_scale_factor_mmhg",
    "epochs",
    "batch_size",
    "optimizer_name",
    "loss",
    "acceptable_error_mg_l",
    "acceptable_fraction_outside_error",
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
            error_thresholds_mg_l={0.5},
            acceptable_error_mg_l=0.5,
            label_scale_factor_mmhg=100,
        )
        assert actual["dataset_filepath"] == sentinel.dataset_filepath
        assert actual["dataset_hash"] == sentinel.dataset_hash

    def test_calculates_dataset_cache_attributes(self, mock_dataset_fns):
        actual = module._calculate_additional_hyperparameters(
            dataset_filename=sentinel.dataset_filename,
            error_thresholds_mg_l={0.5},
            acceptable_error_mg_l=0.5,
            label_scale_factor_mmhg=100,
        )
        assert actual["dataset_filepath"] == sentinel.dataset_filepath
        mock_dataset_fns["mock_get_dataset_csv_hash"].assert_has_calls(
            [call(sentinel.dataset_filepath)]
        )

    def test_calculates_metrics(self, mock_dataset_fns):
        actual = module._calculate_additional_hyperparameters(
            dataset_filename=sentinel.dataset_filename,
            error_thresholds_mg_l={0.1, 0.3},
            acceptable_error_mg_l=0.5,
            label_scale_factor_mmhg=100,
        )
        actual_metric_names = {
            metric.__name__ if hasattr(metric, "__name__") else metric
            for metric in actual["metrics"]
        }
        expected_metric_names = {
            "mean_squared_error",
            "mean_absolute_error",
            "fraction_outside_0_1_mg_l_error",
            "fraction_outside_0_3_mg_l_error",
            "fraction_outside_0_5_mg_l_error",
        }
        assert actual_metric_names == expected_metric_names


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


@pytest.mark.parametrize(
    "dictionary,expected_output",
    (
        ({}, {}),
        ({"key1": None}, {}),
        ({"key1": False}, {"key1": False}),
        ({"key1": "value", "key2": None}, {"key1": "value"}),
    ),
)
def test_remove_items_with_no_value(dictionary, expected_output):
    assert module._remove_items_with_no_value(dictionary) == expected_output


@pytest.fixture
def mock_optimizer(mocker):
    mock_optimizer_class = Mock()

    class MockOptimizer(Enum):
        MOCK_OPTIMIZER = mock_optimizer_class

    mocker.patch.object(module, "Optimizer", MockOptimizer)

    return mock_optimizer_class


class TestGetOptimizer:
    def test_gets_correct_optimizer(self, mock_optimizer):
        hyperparameters = {"optimizer_name": "mock_optimizer"}

        module.get_optimizer(hyperparameters)
        mock_optimizer.assert_called_with()

    def test_specifies_learning_rate(self, mock_optimizer):
        hyperparameters = {
            "optimizer_name": "mock_optimizer",
            "learning_rate": sentinel.learning_rate,
        }

        module.get_optimizer(hyperparameters)
        mock_optimizer.assert_called_with(lr=sentinel.learning_rate)

    def test_raises_on_unknown_optimizer_name(self, mocker):
        hyperparameters = {"optimizer_name": "daniel day lewis"}

        with pytest.raises(module.UnknownOptimizerName):
            module.get_optimizer(hyperparameters)

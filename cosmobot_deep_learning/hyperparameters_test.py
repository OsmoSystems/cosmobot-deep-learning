from unittest.mock import call, sentinel

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
    "optimizer",
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

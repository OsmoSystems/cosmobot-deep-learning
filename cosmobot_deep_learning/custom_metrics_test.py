from unittest.mock import Mock, sentinel

import pytest

from . import custom_metrics as module


class TestGetFractionOutsideErrorThresholdFn:
    def test_sets_new_name(self):
        actual_fn = module.get_fraction_outside_error_threshold_fn(
            error_threshold_mg_l=0.5, label_scale_factor_mmhg=100
        )

        assert actual_fn.__name__ == "fraction_outside_0_5_mg_l_error"


class TestGetFractionOutsideErrorThresholdFnName:
    def test_generates_correct_name(self):
        actual_fn_name = module._get_fraction_outside_error_threshold_fn_name(
            error_threshold_mg_l=0.5
        )

        assert actual_fn_name == "fraction_outside_0_5_mg_l_error"


class TestFunctionNamify:
    @pytest.mark.parametrize(
        "input,expected",
        [
            # fmt: off
            (12.3, "12_3"),
            (0.123, "0_12"),
            (0.5, "0_5"),
            (.3, "0_3")
            # fmt: on
        ],
    )
    def test_function_namify(self, input, expected):
        actual = module._function_namify(input)
        assert actual == expected


class TestRestoreBestWeights:
    def test_calls_set_weights_with_best_weights(self):
        metric = "some_metric"
        epoch_metric_values = [0.3, 0.1, 0.15]
        epoch_weights = list(range(3))

        mock_model = Mock()
        mock_model.get_weights.side_effect = epoch_weights

        callback = module.RestoreBestWeights(metric)
        callback.set_model(mock_model)
        for value in epoch_metric_values:
            callback.on_epoch_end(sentinel.epoch, {metric: value})
        callback.on_train_end()

        assert callback.best_value == 0.1
        mock_model.set_weights.assert_called_once_with(1)

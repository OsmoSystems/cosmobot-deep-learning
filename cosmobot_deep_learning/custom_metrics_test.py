from unittest.mock import Mock, sentinel, call

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


@pytest.fixture
def mock_log_predictions_callback_methods(mocker):
    mocker.patch.object(module.LogPredictionsAndWeights, "_log_predictions")
    mocker.patch.object(module.LogPredictionsAndWeights, "_log_predictions_chart")
    mocker.patch.object(module.LogPredictionsAndWeights, "_rename_model_best")
    mocker.patch.object(module.LogPredictionsAndWeights, "_save_final_weights")


@pytest.fixture
def mock_model(mocker):
    mock_model = Mock()
    mock_model.get_weights.return_value = sentinel.weights
    return mock_model


class TestLogPredictionsAndWeights:
    def test_calls_get_weights_when_best_updates(
        self, mock_log_predictions_callback_methods, mock_model
    ):
        metric = "some_metric"

        callback = module.LogPredictionsAndWeights(
            metric,
            (sentinel.x_train, sentinel.y_train, sentinel.x_dev, sentinel.y_dev),
            label_scale_factor_mmhg=1,
        )
        epoch_metric_values = [0.3, 0.1, 0.15]
        callback.set_model(mock_model)

        for i, value in enumerate(epoch_metric_values):
            callback.on_epoch_end(i, {metric: value})

        # Called
        # once to initialize best_weights,
        # once to predict on epoch 0 (0 % 10 == 0)
        # once to update best_weights when improving to 0.1
        assert mock_model.get_weights.call_count == 3

    @pytest.mark.parametrize(
        ("metrics,log_prediction_epochs"),
        [
            (
                [0.3, 0.1, 0.15],  # Epoch 1 is best
                [1],  # Log predictions only on best epoch
            ),
            (
                [0.3, 0.15, 0.1, 0.15],  # Improves until Epoch 2
                [2],  # Log predictions only at end of improvement streak
            ),
            (
                [0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.3],  # Epoch 5 is best
                [5],  # Only logs epoch 5 once, despite being best + on interval
            ),
            ([0.3, 0.15, 0.3, 0.3, 0.1, 0.3], [1, 4]),  # Improves at epochs 1 and 4
            (
                [
                    0.3,
                    0.15,
                    0.3,
                    0.16,
                    0.3,
                    0.1,
                    0.3,
                ],  # Improves (globally) at epochs 1 and 5
                [1, 5],  # Ignores local improvement at epoch 3
            ),
        ],
    )
    def test_logs_predictions_and_weights_as_expected(
        self,
        metrics,
        log_prediction_epochs,
        mock_log_predictions_callback_methods,
        mock_model,
    ):
        metric = "some_metric"

        callback = module.LogPredictionsAndWeights(
            metric,
            (sentinel.x_train, sentinel.y_train, sentinel.x_dev, sentinel.y_dev),
            label_scale_factor_mmhg=1,
            epoch_interval=5,
        )

        callback.set_model(mock_model)
        callback.params = {"epochs": len(metrics)}

        for i, value in enumerate(metrics):
            callback.on_epoch_end(i, {metric: value})

        callback.on_train_end({metric: value})

        # Assert log_predictions was called during all of the expected epochs
        expected_log_prediction_calls = [
            call(sentinel.weights, epoch) for epoch in log_prediction_epochs
        ]

        # fmt: off
        # Disable black to keep type ignore comments on one line
        # disable mypy to ignore errors about missing definitions for mock attributes
        callback._log_predictions.assert_has_calls(expected_log_prediction_calls)  # type: ignore

        expected_improvement_streak_count = len(log_prediction_epochs)
        # Model is renamed and charts are uploaded at the end of every improvement streak
        # An additional chart is uploaded at the end of training
        assert callback._log_predictions_chart.call_count == expected_improvement_streak_count + 1  # type: ignore
        assert callback._rename_model_best.call_count == expected_improvement_streak_count  # type: ignore
        # fmt: on

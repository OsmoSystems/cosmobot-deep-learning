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

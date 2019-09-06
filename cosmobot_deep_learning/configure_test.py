from typing import List

import pytest

from . import configure as module


class TestParseArgs:
    def test_all_args_parsed_appropriately(self):
        args_in = [
            "--no-gpu",
            "--dryrun",
            "--dataset-cache",
            "10k-images-and-temp",
            "--epochs",
            "100",
            "--optimizer-name",
            "adam",
            "--learning-rate",
            "0.001",
        ]

        expected_args_out = {
            "no_gpu": True,
            "dryrun": True,
            "dataset_cache_name": "10k-images-and-temp",
            "epochs": 100,
            "optimizer_name": "adam",
            "learning_rate": 0.001,
        }

        assert vars(module.parse_model_run_args(args_in)) == expected_args_out

    def test_optional_args_take_default_value(self):
        args_in: List[str] = []
        assert vars(module.parse_model_run_args(args_in))["dataset_cache_name"] is None

    def test_unrecognized_args_throws(self):
        args_in = ["--extra"]
        with pytest.raises(SystemExit):
            module.parse_model_run_args(args_in)

    @pytest.mark.parametrize(
        "args_in,expected_no_gpu_value",
        (
            ([], False),
            (["--no-gpu"], True),
            (["--no-gpu=True"], True),
            (["--no-gpu=False"], False),
            (["--no-gpu", "True"], True),
            (["--no-gpu", "False"], False),
        ),
    )
    def test_no_gpu_formats(self, args_in, expected_no_gpu_value):
        args_out = vars(module.parse_model_run_args(args_in))
        assert args_out["no_gpu"] == expected_no_gpu_value

    @pytest.mark.parametrize(
        "args_in,expected_dryrun_value",
        (
            ([], False),
            (["--dryrun"], True),
            (["--dryrun=True"], True),
            (["--dryrun=False"], False),
            (["--dryrun", "True"], True),
            (["--dryrun", "False"], False),
        ),
    )
    def test_dryrun_formats(self, args_in, expected_dryrun_value):
        args_out = vars(module.parse_model_run_args(args_in))
        assert args_out["dryrun"] == expected_dryrun_value


class TestGetModelNameFromFilepath:
    @pytest.mark.parametrize(
        "filepath,expected_name",
        [
            ("/path/to/model.py", "model"),
            ("/path/to/my_model.py", "my_model"),
            ("model.py", "model"),
            ("model", "model"),
            (__file__, "configure_test"),  # meta
        ],
    )
    def test_get_model_name_from_filepath(self, filepath, expected_name):
        actual = module.get_model_name_from_filepath(filepath)
        assert actual == expected_name


@pytest.mark.parametrize(
    "value,expected",
    [
        ("False", False),
        ("True", True),
        ("false", False),
        ("true", True),
        (False, False),
        (True, True),
    ],
)
def test_str_to_bool(value, expected):
    assert module._string_to_bool(value) == expected

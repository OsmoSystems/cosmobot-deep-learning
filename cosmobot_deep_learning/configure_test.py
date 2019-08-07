from typing import List

import pytest

from . import configure as module


class TestParseArgs:
    def test_all_args_parsed_appropriately(self):
        args_in = ["--gpu", "-1"]

        expected_args_out = {"gpu": -1}

        assert vars(module.parse_model_run_args(args_in)) == expected_args_out

    def test_missing_required_args_throws(self):
        args_in: List = []
        with pytest.raises(SystemExit):
            module.parse_model_run_args(args_in)

    def test_unrecognized_args_throws(self):
        args_in = ["--extra"]
        with pytest.raises(SystemExit):
            module.parse_model_run_args(args_in)


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

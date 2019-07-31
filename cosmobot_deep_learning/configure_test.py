from typing import List

import pytest

from . import configure as module


class TestParseArgs:
    def test_all_args_parsed_appropriately(self):
        args_in = ["--gpu", "-1", "--batch-size", "10", "--epochs", "1000"]

        expected_args_out = {"gpu": -1, "batch_size": 10, "epochs": 1000}

        assert vars(module.parse_args(args_in)) == expected_args_out

    def test_missing_required_args_throws(self):
        args_in: List = []
        with pytest.raises(SystemExit):
            module.parse_args(args_in)

    def test_unrecognized_args_throws(self):
        args_in = ["--extra"]
        with pytest.raises(SystemExit):
            module.parse_args(args_in)

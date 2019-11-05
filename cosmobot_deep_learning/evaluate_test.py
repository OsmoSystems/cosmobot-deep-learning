import os
from unittest.mock import Mock, sentinel

import pytest

from . import evaluate as module


@pytest.mark.parametrize(
    "args,expected_parsed_args",
    (
        (
            ["run_123", "dataset_123", "-c", "column_a"],
            {
                "wandb_run_id": "run_123",
                "dataset_filename": "dataset_123",
                "sampling_column_name": "column_a",
            },
        ),
        (
            ["run_123", "dataset_123"],
            {
                "wandb_run_id": "run_123",
                "dataset_filename": "dataset_123",
                "sampling_column_name": None,
            },
        ),
    ),
)
def test_argument_parser(args, expected_parsed_args):
    parsed_args = module._parse_args(args)
    assert vars(parsed_args) == expected_parsed_args


def test_get_run_path(mocker):
    run_id = "run_123"
    wandb_settings = {
        "entity": "ozmoh",
        "project": "measure-the-do",
    }
    expected_run_path = "ozmoh/measure-the-do/run_123"

    mocker.patch.object(module, "_load_wandb_settings", return_value=wandb_settings)

    mock_api = Mock()
    mock_wandb = mocker.patch.object(module, "wandb")
    mock_wandb.Api = Mock(return_value=mock_api)

    module._get_run(run_id)

    mock_api.run.assert_called_once_with(expected_run_path)


def test_model_best_h5_file(mocker):
    mock_temp_dir = Mock()
    mock_temp_dir.name = "mock_temp_dir"
    mock_tempfile = mocker.patch.object(module, "tempfile")
    mock_tempfile.TemporaryDirectory.return_value = mock_temp_dir

    mocker.patch.object(module, "_download_run_file")

    with module.ModelBestH5File(sentinel.run) as local_filename:
        assert local_filename == os.path.join("mock_temp_dir", "model-best.h5")

    # make sure it cleaned up the temporary directory
    assert mock_temp_dir.cleanup.was_called_once()

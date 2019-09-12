import os
import textwrap

import pandas as pd
import pytest

from cosmobot_deep_learning.constants import AUTO_ASSIGN_GPU

from . import gpu as module


@pytest.fixture
def mock_get_gpus_free_memory(mocker):
    return mocker.patch.object(module, "get_gpus_free_memory")


@pytest.fixture
def mock_os_environ(mocker):
    return mocker.patch.object(os, "environ", {})


class TestGetGPUFreeMemory:
    def test_parses_memory_string_to_int(self, mocker):
        # fmt: off
        mock_nvidia_smi_result = textwrap.dedent(
            """ index, memory.free [MiB]
                0, 9999 MiB
                1, 1 MiB
            """
        )
        # fmt: on

        mocker.patch.object(
            module, "_query_nvidia_smi_free_memory", return_value=mock_nvidia_smi_result
        )

        expected = pd.Series([9999, 1], index=[0, 1], name="Memory Free (MiB)")
        expected.index.name = "GPU ID"

        actual = module.get_gpus_free_memory()

        pd.testing.assert_series_equal(actual, expected)


class TestSetCUDAVisibleDevices:
    @pytest.mark.parametrize(
        "stats,expected",
        (
            ([7001], "0"),
            ([7001, 10], "0"),
            ([10, 7001], "1"),
            ([10, 10, 7001], "2"),
            ([10, 10, 10, 7001], "3"),
            ([10, 5000, 10, 7001], "3"),
        ),
    )
    def test_gets_gpu_with_most_free_memory(
        self, mock_get_gpus_free_memory, mock_os_environ, stats, expected
    ):
        mock_memory_stats = pd.Series(stats)
        mock_get_gpus_free_memory.return_value = mock_memory_stats

        module.set_cuda_visible_devices(gpu=AUTO_ASSIGN_GPU)

        assert mock_os_environ["CUDA_VISIBLE_DEVICES"] == expected

    @pytest.mark.parametrize("gpu_value", ("-1", "2", "0,3"))
    def test_sets_gpu_when_specified(self, mock_os_environ, gpu_value):
        module.set_cuda_visible_devices(gpu=gpu_value)

        assert mock_os_environ["CUDA_VISIBLE_DEVICES"] == gpu_value

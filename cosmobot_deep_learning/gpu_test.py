import os

import pandas as pd
import pytest

from . import gpu as module


@pytest.fixture
def mock_get_gpus_free_memory(mocker):
    return mocker.patch.object(module, "get_gpus_free_memory")


@pytest.fixture
def mock_os_environ(mocker):
    return mocker.patch.object(os, "environ")


class TestGetGPUFreeMemory:
    def test_parses_memory_string_to_int(self, mocker):
        # fmt: off
        mock_nvidia_smi_result = (
            "index, memory.free [MiB]\n"
            "0, 9999 MiB\n"
            "1, 1 MiB"
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
    def test_sets_cpu_mode_on_dryrun(self, mock_os_environ):
        module.set_cuda_visible_devices(gpu="1", dryrun=True)

        mock_os_environ.__setitem__.assert_called_with("CUDA_VISIBLE_DEVICES", "-1")

    def test_sets_cpu_mode_when_gpu_disabled(self, mock_os_environ):
        module.set_cuda_visible_devices(gpu="no-gpu", dryrun=False)

        mock_os_environ.__setitem__.assert_called_with("CUDA_VISIBLE_DEVICES", "-1")

    @pytest.mark.parametrize(
        "stats,expected",
        (
            ({"GPU ID": [0], "Memory Free (MiB)": [7001]}, "0"),
            ({"GPU ID": [0, 1], "Memory Free (MiB)": [7001, 10]}, "0"),
            ({"GPU ID": [0, 1], "Memory Free (MiB)": [10, 7001]}, "1"),
            ({"GPU ID": [0, 1, 2], "Memory Free (MiB)": [10, 10, 7001]}, "2"),
            ({"GPU ID": [0, 1, 2, 3], "Memory Free (MiB)": [10, 10, 10, 7001]}, "3"),
            ({"GPU ID": [0, 1, 2, 3], "Memory Free (MiB)": [10, 5000, 10, 7001]}, "3"),
        ),
    )
    def test_gets_first_available_gpu(
        self, mock_get_gpus_free_memory, mock_os_environ, stats, expected
    ):
        mock_memory_stats = pd.DataFrame(stats).set_index("GPU ID")["Memory Free (MiB)"]
        mock_get_gpus_free_memory.return_value = mock_memory_stats

        module.set_cuda_visible_devices(gpu="auto", dryrun=False)

        mock_os_environ.__setitem__.assert_called_with("CUDA_VISIBLE_DEVICES", expected)

    def test_raises_if_no_available_gpu(self, mock_get_gpus_free_memory):
        mock_memory_stats = pd.Series(
            [module.GPU_AVAILABLE_MEMORY_THRESHOLD - 1], index=[0]
        )
        mock_get_gpus_free_memory.return_value = mock_memory_stats

        with pytest.raises(module.NoGPUAvailable):
            module.set_cuda_visible_devices(gpu="auto", dryrun=False)

    @pytest.mark.parametrize("gpu_value", ("2", "0,3"))
    def test_sets_gpu_when_specified(self, mock_os_environ, gpu_value):
        module.set_cuda_visible_devices(gpu=gpu_value, dryrun=False)

        mock_os_environ.__setitem__.assert_called_with(
            "CUDA_VISIBLE_DEVICES", gpu_value
        )

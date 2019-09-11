import io
import os
import logging
import subprocess

import pandas as pd

GPU_AVAILABLE_MEMORY_THRESHOLD = 4000


class NoGPUAvailable(Exception):
    # Raised when no GPUs are available for training and --no-gpu or --dryrun were not set
    pass


def _query_nvidia_smi_free_memory():
    return subprocess.check_output(
        'nvidia-smi --query-gpu="index,memory.free" --format=csv', shell=True
    ).decode("utf-8")


def get_gpus_free_memory() -> pd.Series:
    """ Parse nvidi-smi output into a Series of GPU device ID-indexed free memory for easier handling
    """
    gpu_stats = pd.read_csv(
        io.StringIO(_query_nvidia_smi_free_memory()),
        header=0,
        names=["GPU ID", "Memory Free (MiB)"],
    ).set_index("GPU ID")

    # Convert values like "7159 MiB" to int -> 7159 and return a series
    # expand=False ensures that a Series is returned when there is a single capture group
    return (
        gpu_stats["Memory Free (MiB)"].str.extract(r"(\d+)", expand=False).astype(int)
    )


def set_cuda_visible_devices(gpu, dryrun):
    """ If `gpu` 'auto,' assign CUDA_VISIBLE_DEVICES to first available GPU, otherwise set to `gpu`.
        If `gpu` is 'no-gpu' or `dryrun` is truthy, set CUDA_VISIBLE_DEVICES to -1 for CPU training.
    """

    if gpu == "no-gpu" or dryrun:
        logging.info("Setting CUDA_VISIBLE_DEVICES to -1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif gpu == "auto":
        gpu_free_memory = get_gpus_free_memory()

        # Get devices where most of the memory is free, sorted with most available GPU first
        free_gpus = gpu_free_memory[
            gpu_free_memory > GPU_AVAILABLE_MEMORY_THRESHOLD
        ].sort_values(ascending=False)

        if not free_gpus.size:
            raise NoGPUAvailable("No GPUs with enough available memory")

        device_id = free_gpus.index[0]

        logging.info(f"Setting CUDA_VISIBLE_DEVICES to {device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

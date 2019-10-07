import io
import os
import logging
import subprocess

import pandas as pd
import tensorflow as tf

from cosmobot_deep_learning.constants import AUTO_ASSIGN_GPU


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


def set_cuda_visible_devices(gpu):
    """ If `gpu` is AUTO_ASSIGN_GPU, assign CUDA_VISIBLE_DEVICES to GPU with most available RAM, otherwise set to `gpu`.
    """

    if gpu == AUTO_ASSIGN_GPU:
        gpu_free_memory = get_gpus_free_memory()

        # Sort devices by amount of free memory and take ID of device with the most
        device_id = gpu_free_memory.sort_values(ascending=False).index[0]

        logging.info(f"Setting CUDA_VISIBLE_DEVICES to {device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    else:
        logging.info(f"Setting CUDA_VISIBLE_DEVICES to {gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def dont_use_all_the_gpu_memory():
    """Set up Keras/TensorFlow to allow multiple models to be trained on one GPU"""
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if not gpus:
        return

    for gpu in gpus:
        # Throws RuntimeError if not set before GPUs have been initialized
        tf.config.experimental.set_memory_growth(gpu, True)

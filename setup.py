#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="cosmobot_deep_learning",
    version="0.0.1",
    author="Osmo Systems",
    author_email="dev@osmobot.com",
    description="Cosmobot deep learning models",
    url="https://www.github.com/osmosystems/cosmobot-deep-learning.git",
    packages=find_packages(),
    # fmt: off
    install_requires=[
        # dill is required by shap but not installed with it
        "dill",
        "keras",
        "keras_drop_block",
        "numpy",
        "opencv-python",
        "pandas",
        "plotly>=4,<5",
        "picamraw",
        "scipy",
        "sklearn",
        "shap",
        # Older versions break multiprocess https://github.com/pytorch/vision/issues/544
        "tqdm>=4.29",
        "wandb",
    ],
    # fmt: on
    extras_require={
        # On macs/Docker/CI we have to install just `tensorflow` (no GPU support)
        # On non-macs, install just `tensorflow-gpu` to get GPU support
        "no-gpu": ["tensorflow"],
        "gpu": ["tensorflow-gpu"],
    },
    include_package_data=True,
)

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
        "keras",
        "keras_resnet",
        "numpy",
        "opencv-python",
        "pandas",
        "plotly>=4,<5",
        "picamraw",
        "scipy",
        # Newer versions of tensorflow have memory issues on our graphics cards
        # https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-456243093
        # On macs we have to install just `tensorflow` (no GPU support)
        # On non-macs, install just `tensorflow-gpu` to get GPU support
        # "tensorflow==1.12.0",
        "tensorflow-gpu==1.12.0",
        "tqdm",
        "wandb",
    ],
    # fmt: on
    include_package_data=True,
)

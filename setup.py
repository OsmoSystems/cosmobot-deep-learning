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
        # Pandas 0.25 breaks tqdm (for now)
        # We can unpin when https://github.com/tqdm/tqdm/issues/780 is fixed
        "pandas <=0.24.2",
        "picamraw",
        "scipy",
        # Newer versions of tensorflow have memory issues on our graphics cards
        # https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-456243093
        "tensorflow-gpu==1.12.0",
        "tqdm",
        "wandb",
    ],
    # fmt: on
    include_package_data=True,
)

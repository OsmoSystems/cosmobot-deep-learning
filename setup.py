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
        "opencv-python",
        # Pandas 0.25 breaks tqdm (for now)
        # We can unpin when https://github.com/tqdm/tqdm/issues/780 is fixed
        "pandas <=0.24.2",
        "picamraw",
        "scipy",
        "tqdm",
    ],
    # fmt: on
    include_package_data=True,
)
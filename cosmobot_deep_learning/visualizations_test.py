import pytest
import numpy as np

from . import visualizations as module


# fmt: off
@pytest.mark.parametrize(
    'array,max_samples,expected_result',
    (
        ([1, 2, 3], 3, [1, 2, 3]),
        ([1, 2, 3], 4, [1, 2, 3]),
        ([1, 2, 3, 4, 5], 3, [1, 3, 5]),
    ),
)
# fmt: on
def test_downsample(array, max_samples, expected_result):
    result = module._downsample(np.array(array), max_samples)
    assert np.array_equal(result, np.array(expected_result))

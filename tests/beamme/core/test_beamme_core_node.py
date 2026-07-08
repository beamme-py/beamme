# The MIT License (MIT)
#
# Copyright (c) 2018-2026 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This script is used to test the functionality of the core node."""

import numpy as np
import pytest

from beamme.core.node import Node


@pytest.mark.parametrize(
    "coordinates",
    (
        np.array([1, 2, 3], dtype=int),
        np.array([1, 2, 3], dtype=float),
        np.array([1.0, 2.5, 3.1], dtype=float),
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        [1.0, 2.5, 3.1],
    ),
)
@pytest.mark.parametrize(
    "increment",
    (
        np.array([2, 3, 4], dtype=int),
        np.array([2, 3, 4], dtype=float),
        np.array([2.0, 3.5, 4.1], dtype=float),
        [2, 3, 4],
        [2.0, 3.0, 4.0],
        [2.0, 3.5, 4.1],
    ),
)
def test_beamme_core_node_coordinates_data_types(
    coordinates, increment, assert_results_close
):
    """Test that different data types for coordinates are handled correctly."""
    node = Node(coordinates=coordinates)
    node.coordinates += increment

    assert_results_close(
        node.coordinates,
        np.array(coordinates, dtype=float) + np.array(increment, dtype=float),
    )

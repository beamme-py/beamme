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
"""Test the performance of the geometric search algorithms."""

import numpy as np
import pytest

from beamme.geometric_search.find_close_points import (
    FindClosePointAlgorithm,
    find_close_points,
)


@pytest.mark.performance
def test_performance_geometric_search_find_close_points_brute_force_cython(
    evaluate_execution_time,
):
    """Test the performance of finding close points using brute force Cython
    algorithm."""

    def repeat_find_random_close_points(n_points, n_runs, algorithm):
        """Repeat finding close points with random points."""
        np.random.seed(seed=1)
        points = np.random.rand(n_points, 3)

        for _ in range(n_runs):
            find_close_points(points, algorithm=algorithm)

    evaluate_execution_time(
        "Find close points (brute force Cython)",
        repeat_find_random_close_points,
        kwargs={
            "n_points": 100,
            "n_runs": 1000,
            "algorithm": FindClosePointAlgorithm.brute_force_cython,
        },
        expected_time=0.025,
    )

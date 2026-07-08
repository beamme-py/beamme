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
"""This file defines the interface to the Cython geometric search functionality."""

import warnings as _warnings

from beamme.geometric_search.utils import cython_is_available as _cython_is_available

if _cython_is_available():
    from beamme.geometric_search.cython_lib import (
        find_close_points as _find_close_points,
    )


def find_close_points_brute_force_cython(
    point_coordinates, tol, *, n_points_performance_warning=5000
):
    """Call the Cython brute force implementation of find close_points."""
    if _cython_is_available():
        n_points = len(point_coordinates)
        if n_points > n_points_performance_warning:
            _warnings.warn(
                "The function find_close_points is called with the brute force algorithm "
                + f"with {n_points} points, for performance reasons other algorithms should be used!"
            )
        return _find_close_points(point_coordinates, tol)
    else:
        raise ModuleNotFoundError(
            "Cython geometric search functionality is not available"
        )

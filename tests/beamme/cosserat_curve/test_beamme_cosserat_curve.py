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
"""Unit tests for the Cosserat curve module."""

import numpy as np
import pytest

from beamme.core.rotation import Rotation


def test_beamme_cosserat_curve_bad_guess_triad(get_cosserat_curve):
    """Check that an error is thrown for a bad guess triad."""
    with pytest.raises(ValueError):
        get_cosserat_curve(
            starting_triad_guess=Rotation([-0.5, 3, -0.5], 2)
            * Rotation([0, 0, 1], np.pi * 0.5),
        )


def test_beamme_cosserat_curve_pvd_series_arguments(get_cosserat_curve):
    """Test the that arguments are correctly processed in the pvd series representation
    of the Cosserat curve."""
    curve = get_cosserat_curve()

    pvd_name = "temp.pvd"

    # Check that errors are raised for invalid argument combinations
    with pytest.raises(
        ValueError, match="The output path must have a .pvd suffix, got .vtu"
    ):
        curve.write_pvd_series("temp.vtu")

    with pytest.raises(
        ValueError,
        match="The keyword arguments 'factors' and 'n_steps' are mutually exclusive.",
    ):
        curve.write_pvd_series(pvd_name, factors=[0.0, 1.0], n_steps=2)

    with pytest.raises(
        ValueError,
        match="One of the keyword arguments 'factors' or 'n_steps' must be provided.",
    ):
        curve.write_pvd_series(pvd_name)


def test_beamme_cosserat_curve_project_point(get_cosserat_curve, assert_results_close):
    """Test that the project point function works as expected."""
    # Load the curve
    curve = get_cosserat_curve()

    # Translate the curve so that the start is at the origin
    curve.translate(-curve.centerline_interpolation(0.0))

    # Check the projection results
    t_ref = 4.264045157204052
    assert_results_close(t_ref, curve.project_point([-5, 1, 1]))
    assert_results_close(t_ref, curve.project_point([-5, 1, 1], t0=2.0))
    assert_results_close(t_ref, curve.project_point([-5, 1, 1], t0=4.0))

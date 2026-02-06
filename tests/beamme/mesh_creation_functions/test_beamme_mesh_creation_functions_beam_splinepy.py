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
"""Unit tests for the splinepy beam mesh creation functions."""

from beamme.mesh_creation_functions.beam_splinepy import (
    get_curve_function_and_jacobian_for_integration,
)


def test_beamme_mesh_creation_functions_beam_splinepy_function_and_jacobian(
    get_splinepy_object, assert_results_close
):
    """Unittest the function and jacobian creation in the
    create_beam_mesh_from_splinepy function."""

    curve = get_splinepy_object("curve_nurbs")
    r, dr, _, _ = get_curve_function_and_jacobian_for_integration(curve, tol=10)

    t_values = [5.0 / 7.0, -0.3, 1.2]
    results_r = [
        [1.4285714285714286, 0.8163265306122449, -0.4081632653061225],
        [-0.6, -1.2, 0.6],
        [2.4, -0.8, 0.4],
    ]
    results_dr = [
        [2.0, -1.7142857142857144, 0.8571428571428572],
        [2.0, 4.0, -2.0],
        [2.0, -4.0, 2.0],
    ]

    for t, result_r, result_dr in zip(t_values, results_r, results_dr):
        assert_results_close(r(t), result_r)
        assert_results_close(dr(t), result_dr)

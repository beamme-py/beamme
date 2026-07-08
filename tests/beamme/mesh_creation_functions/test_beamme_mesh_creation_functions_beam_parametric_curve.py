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
"""Unit tests for the parametric beam mesh creation functions."""

import re

import numpy as np
import pytest

from beamme.mesh_creation_functions.beam_parametric_curve import (
    _ArcLengthEvaluation,
    create_beam_mesh_parametric_curve,
)


@pytest.mark.parametrize(
    "arc_length_evaluation_kwargs,ref_results",
    (
        (
            # No special kwargs, do the default integration
            {"scipy_integrate": False},
            {
                "arc_length": 19.205966821352146,
                "t_grid": [0.0, 3.3333333333333335, 6.666666666666667, 10.0],
                "S_grid": [
                    0.0,
                    6.243182304721144,
                    12.486364609442289,
                    19.205966821352146,
                ],
                "S_from_t": 6.4879736861022,
                "t_from_S": 2.7290710424527393,
            },
        ),
        (
            # Use scipy's integration to get an approximation for the integration intervals
            {"scipy_integrate": True},
            {
                "arc_length": 19.999999999702904,
                "t_grid": [
                    0.0,
                    1.6666666666666667,
                    3.3333333333333335,
                    5.0,
                    5.416666666666667,
                    5.833333333333333,
                    6.25,
                    6.256510416666667,
                    6.263020833333333,
                    6.26953125,
                    6.272786458333333,
                    6.276041666666667,
                    6.279296875,
                    6.280110677083333,
                    6.280924479166667,
                    6.28173828125,
                    6.282552083333333,
                    6.283365885416667,
                    6.2841796875,
                    6.285807291666667,
                    6.287434895833333,
                    6.2890625,
                    6.302083333333333,
                    6.315104166666667,
                    6.328125,
                    6.354166666666667,
                    6.380208333333333,
                    6.40625,
                    6.458333333333333,
                    6.510416666666667,
                    6.5625,
                    6.666666666666667,
                    6.770833333333333,
                    6.875,
                    7.083333333333333,
                    7.291666666666667,
                    7.5,
                    8.333333333333334,
                    9.166666666666666,
                    10.0,
                ],
                "S_grid": [
                    0.0,
                    4.315249744288403,
                    8.477046855746924,
                    10.716337814536773,
                    11.186931450507602,
                    11.766787686452636,
                    12.5005505817755,
                    12.513376587229502,
                    12.526244962830631,
                    12.539155715190516,
                    12.54562698421434,
                    12.552108849027828,
                    12.558601309942881,
                    12.560226080838081,
                    12.561851514003909,
                    12.563477609441836,
                    12.565104375880331,
                    12.566731731113707,
                    12.568359374702904,
                    12.571614583036238,
                    12.57486979136957,
                    12.578124999702904,
                    12.60416666636957,
                    12.630208333036238,
                    12.656249999702904,
                    12.708333333036238,
                    12.76041666636957,
                    12.812499999702904,
                    12.91666666636957,
                    13.020833333036238,
                    13.124999999702904,
                    13.333333333036238,
                    13.54166666636957,
                    13.749999999702904,
                    14.16666666636957,
                    14.583333333036238,
                    14.999999999702904,
                    16.666666666369572,
                    18.333333333036236,
                    19.999999999702904,
                ],
                "S_from_t": 8.720825327609772,
                "t_from_S": 1.9133949225780038,
            },
        ),
        (
            # Use scipy's integration to get an approximation for the integration intervals,
            # but also explicitly provide points where there are kinks in the
            # Jacobian.
            {"scipy_integrate": True, "scipy_integrate_points": [2.0 * np.pi]},
            {
                "arc_length": 20.0,
                "t_grid": [
                    0.0,
                    2.0943951023931953,
                    4.1887902047863905,
                    6.283185307179586,
                    7.5221235381197245,
                    8.761061769059863,
                    10.0,
                ],
                "S_grid": [
                    0.0,
                    5.35587619285631,
                    9.83643789466018,
                    12.566370614359172,
                    15.044247076239449,
                    17.522123538119725,
                    20.0,
                ],
                "S_from_t": 8.546431292379033,
                "t_from_S": 2.021878584190861,
            },
        ),
    ),
)
def test_beamme_mesh_creation_functions_beam_parametric_curve_arc_length_evaluation(
    arc_length_evaluation_kwargs, ref_results, assert_results_close
):
    """Unit tests for the arc length integrator."""

    def function_derivative(t):
        """A C0 continuous function to test the arc length integration.

        The first part is a sinus, and then constant. Thus the analytical integral is
        easy to compute.
        """
        if t < 2.0 * np.pi:
            return [2.0 + np.sin(t), 0.0, 0.0]
        else:
            return [2.0, 0.0, 0.0]

    def function_derivative_vectorized(t):
        """A vectorized wrapper for the previous function."""
        result = []
        for t_i in t:
            result.append(function_derivative(t_i))
        return np.array(result)

    interval = (0.0, 10.0)
    n_precomputed_intervals = 3
    t_eval = 3.4657123
    S_eval = 5.1234556

    arc_length_evaluator = _ArcLengthEvaluation(
        function_derivative_vectorized,
        interval=interval,
        n_precomputed_intervals=n_precomputed_intervals,
        **arc_length_evaluation_kwargs,
    )
    assert_results_close(
        arc_length_evaluator.approximate_total_arc_length(), ref_results["arc_length"]
    )
    assert_results_close(arc_length_evaluator.t_grid, ref_results["t_grid"])
    assert_results_close(arc_length_evaluator.S_grid, ref_results["S_grid"])
    assert_results_close(arc_length_evaluator.S_from_t(t_eval), ref_results["S_from_t"])
    assert_results_close(arc_length_evaluator.t_from_S(S_eval), ref_results["t_from_S"])


def test_beamme_mesh_creation_functions_beam_parametric_curve_arc_length_evaluation_arguments():
    """Check that an error is raised for wrong combination of arguments."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "scipy_integrate_points cannot be provided if scipy_integrate is False!"
        ),
    ):
        _ArcLengthEvaluation(
            None, None, scipy_integrate=False, scipy_integrate_points=[0.5]
        )


def test_beamme_mesh_creation_functions_beam_parametric_curve_interval():
    """Check that an error is raised if wrong intervals are given."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Interval must be a 1D sequence of exactly two values, got array with shape (1, 2)."
        ),
    ):
        create_beam_mesh_parametric_curve(None, None, None, None, interval=[[0, 1]])

    with pytest.raises(
        ValueError,
        match=re.escape("Interval must contain exactly two values, got 3."),
    ):
        create_beam_mesh_parametric_curve(None, None, None, None, interval=[0, 1, 2])

    with pytest.raises(
        ValueError,
        match=re.escape("Interval must be in ascending order, got [1. 0.]."),
    ):
        create_beam_mesh_parametric_curve(None, None, None, None, interval=[1, 0])


def test_beamme_mesh_creation_functions_beam_parametric_curve_vectorized():
    """Check that an error is raised if wrong combination of vectorized functions is
    given."""

    def function_vectorized(dummy):
        """A dummy function for testing."""
        pass

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Function derivative could not be determined! For vectorized inputs, "
            "the function derivative must be explicitly provided."
        ),
    ):
        create_beam_mesh_parametric_curve(
            None,
            None,
            None,
            function=function_vectorized,
            interval=(0, 1),
            vectorized=True,
        )

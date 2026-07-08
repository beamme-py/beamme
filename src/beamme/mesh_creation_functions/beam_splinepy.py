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
"""Create a beam filament from a curve represented with splinepy."""

import numpy as _np

from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve as _create_beam_mesh_parametric_curve,
)


def get_curve_function_and_jacobian_for_integration(curve):
    """Return function objects for evaluating the curve and the derivative.

    These functions are used in the curve integration.

    Args
    ----
    curve: splinepy object
        Curve that is used to describe the beam centerline.

    Return
    ----
    (function, jacobian, curve_start, curve_end):
        function:
            Function for evaluating a position on the curve
        jacobian:
            Function for evaluating the tangent along the curve
        curve_start:
            Parameter coordinate for the start for the curve
        curve_end:
            Parameter coordinate for the end for the curve
    """
    curve_start = curve.parametric_bounds[0][0]
    curve_end = curve.parametric_bounds[1][0]

    def eval_r(t):
        """Evaluate the position along the curve.

        We need to pass an array with shape (n, 1) to the splinepy function, we do this
        by passing a reordered view of t.
        """
        t = _np.asarray(t)
        return curve.evaluate(t[:, None])

    def eval_rp(t):
        """Evaluate the derivative along the curve.

        We need to pass an array with shape (n, 1) to the splinepy function, we do this
        by passing a reordered view of t.
        """
        t = _np.asarray(t)
        return curve.derivative(t[:, None], orders=[1])

    return eval_r, eval_rp, curve_start, curve_end


def create_beam_mesh_from_splinepy(mesh, beam_class, material, curve, **kwargs):
    """Generate a beam from a splinepy curve.

    Args
    ----
    mesh: Mesh
        Mesh that the curve will be added to.
    beam_class: Beam
        Class of beam that will be used for this line.
    material: Material
        Material for this line.
    curve: splinepy object
        Curve that is used to describe the beam centerline.

    **kwargs (for all of them look into create_beam_mesh_function)
    ----
    n_el: int
        Number of equally spaced beam elements along the line. Defaults to 1.
        Mutually exclusive with l_el.
    l_el: float
        Desired length of beam elements. Mutually exclusive with n_el.
        Be aware, that this length might not be achieved, if the elements are
        warped after they are created.

    Return:
        Return value from create_beam_mesh_function
    """
    (
        function,
        jacobian,
        curve_start,
        curve_end,
    ) = get_curve_function_and_jacobian_for_integration(curve)

    # Create the beams
    return _create_beam_mesh_parametric_curve(
        mesh,
        beam_class,
        material,
        function,
        [curve_start, curve_end],
        function_derivative=jacobian,
        vectorized=True,
        **kwargs,
    )

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
"""Test the performance of the parametric curve creation function."""

import numpy as np
import pytest

from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)
from beamme.mesh_creation_functions.beam_splinepy import create_beam_mesh_from_splinepy


@pytest.mark.performance
def test_performance_mesh_creation_functions_beam_parametric_curve(
    evaluate_execution_time, get_parametric_function
):
    """Test the performance of the parametric curve creation function."""

    mesh = Mesh()
    material = MaterialBeamBase()

    evaluate_execution_time(
        "BeamMe: mesh_creation_functions: Parametric curve",
        create_beam_mesh_parametric_curve,
        args=(
            mesh,
            Beam3,
            material,
            get_parametric_function("distorted_helix"),
            [0, 2 * np.pi],
        ),
        kwargs={"n_el": 500},
        expected_time=0.35,
    )


@pytest.mark.performance
def test_performance_mesh_creation_functions_beam_parametric_curve_splinepy(
    get_splinepy_object,
    evaluate_execution_time,
):
    """Test the performance of the parametric curve creation function with a
    splinepy object.

    Compared to a pure given function, this method does not have the
    overhead of evaluating the Jacobian with automated differentiation,
    as splinepy provides the derivatives explicitly.
    """

    mesh = Mesh()
    material = MaterialBeamBase()

    nurbs = get_splinepy_object("curve_nurbs_distorted")

    def function_call_multiple(*args, **kwargs):
        """Create the mesh multiple times to get a better assessment of the
        performance."""
        for _ in range(20):
            create_beam_mesh_from_splinepy(
                *args,
                **kwargs,
            )

    evaluate_execution_time(
        "BeamMe: mesh_creation_functions: Beam from splinepy",
        function_call_multiple,
        args=(mesh, Beam3, material, nurbs),
        kwargs={"n_el": 50},
        expected_time=0.55,
    )

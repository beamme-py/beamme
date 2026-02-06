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
"""This script is used to test the create parametric curve mesh creation
functions."""

import autograd.numpy as npAD
import numpy as np
import pytest
from autograd import jacobian

from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)

PARAMETRIC_CURVE_TEST_PARAMETRIZATION: tuple = (
    "name,arc_length_integrator_kwargs",
    (
        ("default", {}),
        ("arc-length", {"method": "arc-length", "n_intervals": 10}),
        ("parametric", {"method": "parametric", "n_intervals": 10}),
        (
            "parametric_consistent",
            {"method": "parametric_consistent_middle_nodes", "n_intervals": 10},
        ),
    ),
)


def test_integration_mesh_creation_functions_beam_parametric_curve_3d_helix(
    get_default_test_beam_material,
    get_parametric_function,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a helix from a parametric curve where the parameter is
    transformed so the arc length along the beam is not proportional to the
    parameter."""

    # Create mesh
    mesh = Mesh()

    # Add material and functions.
    mat = get_default_test_beam_material(material_type="reissner")

    # Get the helix curve function
    R = 2.0
    tz = 4.0  # incline
    n = 1  # number of turns
    n_el = 5
    helix = get_parametric_function(
        "helix", R, tz, transformation_factor=2.0, number_of_turns=n
    )

    helix_set = create_beam_mesh_parametric_curve(
        mesh, Beam3rHerm2Line3, mat, helix, [0.0, 2.0 * np.pi * n], n_el=n_el
    )
    mesh.add(helix_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


@pytest.mark.parametrize(*PARAMETRIC_CURVE_TEST_PARAMETRIZATION)
def test_integration_mesh_creation_functions_beam_parametric_curve_3d_helix_length(
    name,
    arc_length_integrator_kwargs,
    get_default_test_beam_material,
    get_parametric_function,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Create a helix from a parametric curve where and check that the correct
    length is returned."""

    mesh_1 = Mesh()
    mesh_2 = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    # Get the helix curve function
    R = 2.0
    tz = 4.0  # incline
    n = 1  # number of turns
    n_el = 3
    helix = get_parametric_function(
        "helix", R, tz, transformation_factor=2.0, number_of_turns=n
    )

    args = [Beam3rHerm2Line3, mat, helix, [0.0, 2.0 * np.pi * n]]
    kwargs = {
        "n_el": n_el,
        "arc_length_integrator_kwargs": arc_length_integrator_kwargs,
    }

    helix_set_1 = create_beam_mesh_parametric_curve(mesh_1, *args, **kwargs)
    mesh_1.add(helix_set_1)

    helix_set_2, length = create_beam_mesh_parametric_curve(
        mesh_2, *args, output_length=True, **kwargs
    )
    mesh_2.add(helix_set_2)

    # Check the computed length
    assert_results_close(
        length, 13.187633539381515 if name == "default" else 13.191216930059845
    )

    # Check that both meshes are equal
    assert_results_close(mesh_1, mesh_2)

    # Compare with reference solution
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=name), mesh_1
    )


@pytest.mark.parametrize(*PARAMETRIC_CURVE_TEST_PARAMETRIZATION)
def test_integration_mesh_creation_functions_beam_parametric_curve_2d_sin(
    name,
    arc_length_integrator_kwargs,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a sin from a parametric curve."""

    # Create mesh
    mesh = Mesh()

    # Add material and functions.
    mat = get_default_test_beam_material(material_type="reissner")

    # Set parameters for the sin.
    n_el = 8

    def sin(t):
        """Parametric curve as a sin."""
        return npAD.array([t, npAD.sin(t)])

    sin_set = create_beam_mesh_parametric_curve(
        mesh,
        Beam3rHerm2Line3,
        mat,
        sin,
        [0.0, 2.0 * np.pi],
        n_el=n_el,
        arc_length_integrator_kwargs=arc_length_integrator_kwargs,
    )
    mesh.add(sin_set)

    # Check the output.
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier=name), mesh
    )


def test_integration_mesh_creation_functions_beam_parametric_curve_3d_rotation(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a line from a parametric curve and prescribe the rotation."""

    # Create mesh
    mesh = Mesh()

    # Add material and functions.
    mat = get_default_test_beam_material(material_type="reissner")

    # Set parameters for the line.
    L = 1.1
    n_el = 4

    def curve(t):
        """Parametric representation of the centerline curve."""
        return npAD.array([L * t, t * t * L * L, 0.0])

    def rotation(t):
        """Function that defines the triad along the centerline curve."""
        rp2 = jacobian(curve)(t)
        rp = [rp2[0], rp2[1], 0]
        R1 = Rotation([1, 0, 0], t * 2 * np.pi)
        R2 = Rotation.from_basis(rp, [0, 0, 1])
        return R2 * R1

    sin_set = create_beam_mesh_parametric_curve(
        mesh,
        Beam3rHerm2Line3,
        mat,
        curve,
        [0.0, 1.0],
        n_el=n_el,
        function_rotation=rotation,
    )
    mesh.add(sin_set)

    # extend test case with different meshing strategy
    create_beam_mesh_parametric_curve(
        mesh,
        Beam3rHerm2Line3,
        mat,
        curve,
        [1.0, 2.5],
        node_positions_of_elements=[0, 1.0 / 1.3, 1.0],
        function_rotation=rotation,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_parametric_curve_3d_line(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a line from a parametric curve."""

    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    def line(t):
        """Create a line with a parametric curve (and a transformed
        parameter)."""
        factor = 2
        t_trans = npAD.exp(factor * t / (2.0 * np.pi)) * t / npAD.exp(factor)
        return npAD.array([t_trans, 0, 0])

    # Create mesh.
    set_1 = create_beam_mesh_parametric_curve(
        mesh,
        Beam3rHerm2Line3,
        mat,
        line,
        [0.0, 5.0],
        node_positions_of_elements=[0, 0.3, 0.5, 1.0],
    )
    mesh.add(set_1)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)

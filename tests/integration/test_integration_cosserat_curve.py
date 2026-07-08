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
"""Integration tests for the Cosserat curve module."""

import numpy as np
import pytest
import pyvista as pv
import quaternion

from beamme.core.element_beam import Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.rotation import Rotation
from beamme.cosserat_curve.warping_along_cosserat_curve import (
    create_transform_boundary_conditions,
    get_mesh_transformation,
    warp_mesh_along_curve,
)
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix


def create_beam_solid_input_file(
    beam_type, material, get_corresponding_reference_file_path
):
    """Create a beam and solid input file for testing purposes."""
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_solid_brick"
        ),
        convert_input_to_mesh=True,
    )

    create_beam_mesh_helix(
        mesh,
        beam_type,
        material,
        [0, 0, 1],
        [0, 0, 0],
        [2, 0, 0],
        helix_angle=0.4,
        turns=3,
        n_el=5,
    )
    return mesh


@pytest.mark.parametrize(
    ("twist_type", "twist_angle", "starting_triad_guess"),
    [
        (None, 0.0, None),
        ("angle", np.pi / 5, None),
        ("starting_rotation", 0.3293917321011114, Rotation([-0.5, 3, -0.5], 2)),
    ],
)
def test_integration_cosserat_curve_translate_and_rotate(
    get_cosserat_curve,
    twist_type,
    twist_angle,
    starting_triad_guess,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that a curve can be loaded, rotated and transformed."""
    if twist_type is None:
        curve = get_cosserat_curve()
    elif twist_type == "angle":
        curve = get_cosserat_curve()
        curve.twist(twist_angle)
    elif starting_triad_guess is not None:
        curve = get_cosserat_curve(starting_triad_guess=starting_triad_guess)
    relative_rotation = Rotation([1, 0, 0], twist_angle)

    # Translate the curve so that the start is at the origin
    curve.translate(-curve.centerline_interpolation(5.0))

    # Rotate the curve around its center point
    origin = [1, 2, 3]
    rotation = Rotation([1, 2, 3], np.pi / 7)
    curve.rotate(rotation, origin=origin)

    # Get the points and rotations at certain points
    t = list(map(float, range(-10, 30, 5)))
    sol_half_pos, sol_half_q = curve.get_centerline_positions_and_rotations(
        t, factor=0.5
    )
    sol_full_pos, sol_full_q = curve.get_centerline_positions_and_rotations(
        t, factor=1.0
    )

    def load_compare(name):
        """Load the compare files and return a numpy array."""
        return np.loadtxt(
            get_corresponding_reference_file_path(
                additional_identifier=name, extension="txt"
            )
        )

    def get_compare_rot_with_twist(name):
        """Apply twist rotations to reference files."""
        rotations = (
            quaternion.from_float_array(load_compare(name))
            * relative_rotation.get_numpy_quaternion()
        )
        return rotations

    # Note: The tolerances are increased due to issues with the spline
    #     interpolation, see `beamme.cosserat_curve.get_spline_interpolation`.
    assert_results_close(sol_half_pos, load_compare("pos_half_ref"), atol=1e-12)
    assert_results_close(sol_half_q, get_compare_rot_with_twist("q_half_ref"))
    assert_results_close(sol_full_pos, load_compare("pos_full_ref"), atol=1e-12)
    assert_results_close(sol_full_q, get_compare_rot_with_twist("q_full_ref"))


def test_integration_cosserat_curve_vtk_representation(
    tmp_path,
    get_cosserat_curve,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the vtk representation of the Cosserat curve."""
    reference_path = get_corresponding_reference_file_path(extension="vtu")
    result_path = tmp_path / reference_path.name

    curve = get_cosserat_curve()
    pv.UnstructuredGrid(curve.get_pyvista_polyline()).save(result_path)

    assert_results_close(reference_path, result_path)


@pytest.mark.parametrize(
    ("factors", "n_steps", "reference_name"),
    [([0.1, 0.7], None, "factors"), (None, 3, "steps")],
)
def test_integration_cosserat_curve_pvd_series(
    factors,
    n_steps,
    reference_name,
    tmp_path,
    get_corresponding_reference_file_path,
    get_cosserat_curve,
    assert_results_close,
):
    """Test the pvd series representation of the Cosserat curve."""
    reference_path = get_corresponding_reference_file_path(
        additional_identifier=reference_name, extension="pvd"
    )
    result_path = tmp_path / reference_path.name

    curve = get_cosserat_curve()
    curve.write_pvd_series(result_path, factors=factors, n_steps=n_steps)

    assert_results_close(reference_path, result_path)
    for i_step in range(len(factors) if factors is not None else n_steps):
        step_reference_path = get_corresponding_reference_file_path(
            additional_identifier=reference_name + "." + str(i_step),
            extension="vtu",
        )
        step_result_path = tmp_path / step_reference_path.name
        assert_results_close(step_reference_path, step_result_path)


def test_integration_cosserat_curve_mesh_transformation(
    get_cosserat_curve,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that the get_mesh_transformation function works as expected."""
    curve = get_cosserat_curve()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation.from_quaternion(rot)
    curve.translate(-pos)
    curve.translate([1, 2, 3])

    mesh = create_beam_solid_input_file(
        Beam3, MaterialBeamBase(), get_corresponding_reference_file_path
    )
    pos, rot = get_mesh_transformation(
        curve,
        mesh.nodes,
        origin=[0.5, 1.0, -1.0],
        reference_rotation=(
            Rotation([0, 0, 1], -0.5 * np.pi) * Rotation([0, 1, 0], -0.5 * np.pi)
        ),
        n_steps=3,
    )

    assert_results_close(
        get_corresponding_reference_file_path(extension="json"),
        {"pos": pos, "rot": quaternion.as_float_array(rot)},
    )


def test_integration_cosserat_curve_mesh_warp(
    get_cosserat_curve,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Warp a balloon along a centerline."""
    # Load the curve
    curve = get_cosserat_curve()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation.from_quaternion(rot)
    curve.translate(-pos)
    curve.translate([1, 2, 3])

    # Warp the mesh. The reference coordinate system is rotated such that z axis is the longitudinal direction,
    # and x and y are the first and second cross-section basis vectors respectively.
    mesh = create_beam_solid_input_file(
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        get_corresponding_reference_file_path,
    )
    warp_mesh_along_curve(
        mesh,
        curve,
        origin=[0.5, 1.0, -1.0],
        reference_rotation=(
            Rotation([0, 0, 1], -0.5 * np.pi) * Rotation([0, 1, 0], -0.5 * np.pi)
        ),
    )
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_cosserat_curve_mesh_warp_transform_boundary_conditions(
    get_cosserat_curve,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the transform boundary creation function."""
    # Load the curve
    curve = get_cosserat_curve()
    pos, rot = curve.get_centerline_position_and_rotation(0)
    rot = Rotation.from_quaternion(rot)
    curve.translate(-pos)
    curve.translate([1, 2, 3])

    # Load the mesh
    mesh = create_beam_solid_input_file(
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        get_corresponding_reference_file_path,
    )

    # Apply the transform boundary conditions
    create_transform_boundary_conditions(
        mesh,
        curve,
        n_steps=3,
        origin=[2, 3, 0.5],
        reference_rotation=(
            Rotation([0, 0, 1], -0.5 * np.pi) * Rotation([0, 1, 0], -0.5 * np.pi)
        ),
    )

    # Note: The tolerances are increased due to issues with the spline
    #     interpolation, see `beamme.cosserat_curve.get_spline_interpolation`.
    assert_results_close(
        get_corresponding_reference_file_path(), mesh, rtol=1e-12, atol=1e-12
    )

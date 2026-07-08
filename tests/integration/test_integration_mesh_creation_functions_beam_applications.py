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
"""This script is used to test the application mesh creation functions."""

import numpy as np

from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rHerm2Line3, get_four_c_reissner_beam
from beamme.mesh_creation_functions.applications.beam_fibers_in_rectangle import (
    create_fibers_in_rectangle,
)
from beamme.mesh_creation_functions.applications.beam_stent import (
    create_beam_mesh_stent,
)
from beamme.mesh_creation_functions.applications.beam_wire import create_wire_fibers


def test_integration_mesh_creation_functions_beam_applications_stent(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the stent creation function."""
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create mesh.
    create_beam_mesh_stent(
        mesh,
        Beam3rHerm2Line3,
        mat,
        0.11,
        0.02,
        5,
        8,
        fac_bottom=0.6,
        fac_neck=0.52,
        fac_radius=0.36,
        alpha=0.47 * np.pi,
        n_el=2,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_applications_fibers_in_rectangle(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_fibers_in_rectangle function."""
    # Create mesh
    mesh = Mesh()

    # Create mesh.
    mat = get_default_test_beam_material(material_type="reissner")
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        0,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        90,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        -90,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        235,
        0.45,
        0.35,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        1,
        4,
        30,
        0.45,
        5,
        fiber_element_length_min=0.2,
    )
    mesh.translate([0, 0, 1])
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        30,
        0.45,
        0.9,
    )
    mesh.translate([0, 0, 1])

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_applications_fibers_in_rectangle_reference_point(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_fibers_in_rectangle function with using the reference_point
    option."""
    # Create mesh
    mesh = Mesh()

    # Create mesh.
    mat = get_default_test_beam_material(material_type="reissner")
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
    )
    reference_point = 0.5 * np.array([4.0, 1.0]) + 0.1 * np.array(
        [-1.0, 1.0]
    ) / np.sqrt(2.0)
    create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
        reference_point=reference_point,
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_applications_fibers_in_rectangle_return_set(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the set returned by the create_fibers_in_rectangle function."""
    # Create mesh
    mesh = Mesh()

    # Create mesh.
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_fibers_in_rectangle(
        mesh,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        4,
        1,
        45,
        0.45,
        0.35,
    )
    mesh.add(beam_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_applications_wire(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the create_wire_fibers function."""
    # Create mesh
    mesh = Mesh()

    # Create two wires with different parameters.
    mat = get_default_test_beam_material(material_type="reissner")
    mesh_1 = Mesh()
    set_1 = create_wire_fibers(
        mesh_1,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        3.0,
        layers=2,
        n_el=2,
        radius=0.05,
    )
    mesh_2 = Mesh()
    set_2 = create_wire_fibers(
        mesh_2,
        get_four_c_reissner_beam(n_nodes=2, is_hermite_centerline=False),
        mat,
        3.0,
        layers=2,
        n_el=2,
        radius=0.1,
    )
    mesh_2.translate([0.0, 1.5, 0.0])
    mesh.add(mesh_1, mesh_2, set_1, set_2)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)

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
"""This script is used to test 4C input file related functionality."""

import numpy as np

from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.element import Element
from beamme.core.geometry_set import GeometryName
from beamme.core.material import Material
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import (
    Beam3eb,
    Beam3rHerm2Line3,
    get_four_c_reissner_beam,
)
from beamme.four_c.input_file import InputFile
from beamme.mesh_creation_functions.beam_arc import create_beam_mesh_arc_segment_2d
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def create_beam_mesh(
    mesh: Mesh,
    beam_type: type[Element],
    material: Material,
    counter: int = 0,
    n_el: int = 2,
) -> GeometryName:
    """Utility function to create a beam mesh for the tests."""
    if beam_type == Beam3eb:
        # For EB beams, we create a straight line.
        return create_beam_mesh_line(
            mesh,
            beam_type,
            material,
            [15.0 * counter, 0.0, 0.0],
            [15.0 * counter, 10.0, 0.0],
            n_el=n_el,
        )
    else:
        # For all other beams, we create an arc segment with a general spatial rotation.
        temp_mesh = Mesh()
        arc_sets = create_beam_mesh_arc_segment_2d(
            temp_mesh,
            beam_type,
            material,
            [15.0 * counter, 0.0, 0.0],
            10.0,
            0.0,
            np.pi / 3,
            n_el=n_el,
        )
        temp_mesh.rotate(Rotation([1, 0, 0], np.pi / 4))
        mesh.add(temp_mesh)
        return arc_sets


def test_integration_four_c_input_file_vtu_element_blocks(
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test the grouping of elements with the same element data and material in
    blocks."""
    # Get materials
    material_1 = get_default_test_beam_material(
        material_type="reissner", interaction_radius=1.0
    )
    material_2 = get_default_test_beam_material(
        material_type="reissner", interaction_radius=2.0
    )
    material_3 = get_default_test_beam_material(
        material_type="reissner", interaction_radius=3.0
    )

    # Get element types
    beam_eb_2 = Beam3eb
    beam_r_2 = get_four_c_reissner_beam(2, is_hermite_centerline=False)
    beam_r_3 = get_four_c_reissner_beam(3, is_hermite_centerline=True)

    counter = 0

    # Create a mesh with multiple filaments that have different combinations of materials and element types.
    mesh = Mesh()

    # Block 0
    create_beam_mesh(mesh, beam_r_3, material_1, counter)
    counter += 1
    create_beam_mesh(mesh, beam_r_3, material_1, counter)
    counter += 1

    # Block 1
    create_beam_mesh(mesh, beam_r_3, material_2, counter)
    counter += 1

    # Block 2
    create_beam_mesh(mesh, beam_r_2, material_3, counter)
    counter += 1

    # Block 3
    create_beam_mesh(mesh, beam_r_2, material_1, counter)
    counter += 1

    # Block 4
    create_beam_mesh(mesh, beam_eb_2, material_1, counter)
    counter += 1

    # Block 5
    create_beam_mesh(mesh, beam_r_2, material_2, counter)
    counter += 1

    # Block 0 again
    create_beam_mesh(mesh, beam_r_3, material_1, counter)
    counter += 1

    # Block 2 again
    create_beam_mesh(mesh, beam_r_2, material_3, counter)
    counter += 1

    # Add the mesh to the input file.
    input_file = InputFile()
    input_file.add(mesh)

    # Create another mesh with another block. Even though this filament has the
    # same element type and material as block 0, it will not be merged with
    # block 0 since we have already dumped the previous mesh.
    mesh = Mesh()
    create_beam_mesh(mesh, beam_r_3, material_1, counter)
    counter += 1
    input_file.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(),
        input_file,
        four_c_input_file_data_format="vtu",
    )


def test_integration_four_c_input_file_vtu_boundary_conditions(
    get_default_test_beam_material,
    get_bc_data,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that boundary conditions are correctly written to the input file in vtu
    format."""
    # Create mesh
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_beam_mesh(mesh, Beam3rHerm2Line3, mat)

    # Add boundary conditions
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            get_bc_data(identifier=1, num_dof=9),
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            get_bc_data(identifier=2, num_dof=9),
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["end"],
            get_bc_data(identifier=3, num_dof=9),
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["line"],
            get_bc_data(identifier=4, num_dof=9),
            bc_type=bme.bc.neumann,
        )
    )

    # Add the mesh to the input file.
    input_file = InputFile()
    input_file.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(),
        input_file,
        four_c_input_file_data_format="vtu",
    )


def test_integration_four_c_input_file_vtu_boundary_conditions_user_defined_section(
    get_default_test_beam_material,
    get_bc_data,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that boundary conditions with user defined sections are correctly written to
    the input file in vtu format."""
    # Create mesh
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_beam_mesh(mesh, Beam3eb, mat)

    # Add boundary conditions
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            get_bc_data(identifier=1, num_dof=6),
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            get_bc_data(identifier=2, num_dof=6),
            bc_type="DESIGN VOL ALE DIRICH CONDITIONS",
        )
    )

    # Add the mesh to the input file.
    input_file = InputFile()
    input_file.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(),
        input_file,
        four_c_input_file_data_format="vtu",
    )


def test_integration_four_c_input_file_vtu_boundary_conditions_named_geometry_sets(
    get_default_test_beam_material,
    get_bc_data,
    get_corresponding_reference_file_path,
    assert_results_close,
):
    """Test that boundary conditions with named geometry sets are correctly written to
    the input file in vtu format."""
    # Create mesh
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    beam_set = create_beam_mesh(mesh, Beam3eb, mat)

    # Name some of the geometry sets
    beam_set["start"].name = "beam_start"
    beam_set["line"].name = "beam_line"

    # Add boundary conditions
    mesh.add(
        BoundaryCondition(
            beam_set["start"],
            get_bc_data(identifier=1, num_dof=6),
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["end"],
            get_bc_data(identifier=2, num_dof=6),
            bc_type=bme.bc.dirichlet,
        )
    )
    mesh.add(
        BoundaryCondition(
            beam_set["line"],
            get_bc_data(identifier=3, num_dof=6),
            bc_type=bme.bc.neumann,
        )
    )

    # Add the mesh to the input file.
    input_file = InputFile()
    input_file.add(mesh)

    assert_results_close(
        get_corresponding_reference_file_path(),
        input_file,
        four_c_input_file_data_format="vtu",
    )

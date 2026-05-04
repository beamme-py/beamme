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
"""This script is used to test vtk writer functionality of BeamMe."""

from unittest.mock import patch

import numpy as np
import vtk

from beamme.core.element_beam import Beam3
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.core.vtk_writer import VTKType, VTKWriter
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.applications.beam_honeycomb import (
    create_beam_mesh_honeycomb,
)
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_integration_core_vtk_writer_write_vtk(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Test the output created by the VTK writer."""

    # Initialize writer.
    writer = VTKWriter()

    # Add poly line.
    indices = writer.add_points([[0, 0, -2], [1, 1, -2], [2, 2, -1]])
    writer.add_cell(vtk.vtkPolyLine, indices)

    # Add quadratic quad.
    cell_data = {}
    cell_data["cell_data_1"] = 3
    cell_data["cell_data_2"] = [66, 0, 1]
    point_data = {}
    point_data["point_data_1"] = [1, 2, 3, 4, 5, -2, -3, 0]
    point_data["point_data_2"] = [
        [0.25, 0, -0.25],
        [1, 0.25, 0],
        [2, 0, 0],
        [2.25, 1.25, 0.5],
        [2, 2.25, 0],
        [1, 2, 0.5],
        [0, 2.25, 0],
        [0, 1, 0.5],
    ]
    indices = writer.add_points(
        [
            [0.25, 0, -0.25],
            [1, 0.25, 0],
            [2, 0, 0],
            [2.25, 1.25, 0.5],
            [2, 2.25, 0],
            [1, 2, 0.5],
            [0, 2.25, 0],
            [0, 1, 0.5],
        ],
        point_data=point_data,
    )
    writer.add_cell(
        vtk.vtkQuadraticQuad, indices[[0, 2, 4, 6, 1, 3, 5, 7]], cell_data=cell_data
    )

    # Add tetrahedron.
    cell_data = {}
    cell_data["cell_data_2"] = [5, 0, 10]
    point_data = {}
    point_data["point_data_1"] = [1, 2, 3, 4]
    indices = writer.add_points(
        [[3, 3, 3], [4, 4, 3], [4, 3, 3], [4, 4, 4]], point_data=point_data
    )
    writer.add_cell(vtk.vtkTetra, indices[[0, 2, 1, 3]], cell_data=cell_data)

    # Before we can write the data to file we have to store the cell and
    # point data in the grid
    writer.complete_data()

    # Write to file.
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / ref_file.name
    writer.write_vtk(vtk_file, binary=False)

    # Compare the vtk files.
    assert_results_close(ref_file, vtk_file)


def test_integration_core_vtk_writer_beam(
    assert_results_close,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Create a sample mesh and check the VTK output."""

    # Create the mesh.
    mesh = Mesh()

    # Add content to the mesh.
    honeycomb_set = create_beam_mesh_honeycomb(
        mesh,
        Beam3,
        get_default_test_beam_material(),
        2.0,
        2,
        3,
        n_el=2,
    )
    mesh.add(honeycomb_set)

    # Write VTK output, with coupling sets."""
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / "test_1_beam.vtu"
    mesh.write_vtk(
        output_name="test_1",
        output_directory=tmp_path,
        binary=False,
        coupling_sets=True,
    )
    # TODO: Replace this test in the future once we have moved to a new vtu output.
    # assert_results_close(ref_file, vtk_file)

    # Write VTK output, without coupling sets."""
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="no_coupling_beam", extension="vtu"
    )
    vtk_file = tmp_path / "test_2_beam.vtu"
    mesh.write_vtk(
        output_name="test_2",
        output_directory=tmp_path,
        binary=False,
        coupling_sets=False,
    )
    # TODO: Replace this test in the future once we have moved to a new vtu output.
    # assert_results_close(ref_file, vtk_file)

    # Write VTK output, with coupling sets and additional points for visualization."""
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="smooth_centerline_beam", extension="vtu"
    )
    vtk_file = tmp_path / "test_3_beam.vtu"
    mesh.write_vtk(
        output_name="test_3",
        coupling_sets=True,
        output_directory=tmp_path,
        binary=False,
        beam_centerline_visualization_segments=3,
    )
    # TODO: Replace this test in the future once we have moved to a new vtu output.
    # assert_results_close(ref_file, vtk_file)


def test_integration_core_vtk_writer_solid(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Import a solid mesh and check the VTK output."""

    # Convert the solid mesh to beamme objects.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_tube"
        ),
        convert_input_to_mesh=True,
    )

    # Write VTK output.
    ref_file = get_corresponding_reference_file_path(extension="vtu")
    vtk_file = tmp_path / "test_solid.vtu"
    mesh.write_vtk(output_name="test", output_directory=tmp_path, binary=False)

    # Compare the vtk files.
    # TODO: Replace this test in the future once we have moved to a new vtu output.
    # assert_results_close(ref_file, vtk_file)


def test_integration_core_vtk_writer_solid_elements(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Import a solid mesh with all solid types and check the VTK output."""

    # Convert the solid mesh to beamme objects.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_multiple_solid_bricks"
        ),
        convert_input_to_mesh=True,
    )

    # Write VTK output.
    ref_file = get_corresponding_reference_file_path(
        additional_identifier="solid", extension="vtu"
    )
    vtk_file = tmp_path / "test_solid.vtu"
    mesh.write_vtk(output_name="test", output_directory=tmp_path, binary=False)

    # Compare the vtk files.
    # TODO: Replace this test in the future once we have moved to a new vtu output.
    # assert_results_close(ref_file, vtk_file)


def test_integration_core_vtk_writer_curve_cell_data(
    assert_results_close,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
    tmp_path,
):
    """Test that when creating a beam, cell data can be given.

    This test also checks, that the nan values in vtk can be explicitly
    given.
    """

    with (
        patch("beamme.core.vtk_writer.VTK_NAN_FLOAT", 69.69),
        patch("beamme.core.vtk_writer.VTK_NAN_INT", 69),
    ):
        # Create the mesh.
        mesh = Mesh()

        # Add content to the mesh.
        mat = get_default_test_beam_material()
        create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [2, 0, 0], n_el=2)
        create_beam_mesh_line(
            mesh,
            Beam3,
            mat,
            [0, 1, 0],
            [2, 1, 0],
            n_el=2,
            vtk_cell_data={"cell_data": (1, VTKType.int)},
        )
        create_beam_mesh_arc_segment_via_rotation(
            mesh,
            Beam3,
            mat,
            [0, 2, 0],
            Rotation([1, 0, 0], np.pi),
            1.5,
            np.pi / 2.0,
            n_el=2,
            vtk_cell_data={"cell_data": (2, VTKType.int), "other_data": 69},
        )

        # Write VTK output, with coupling sets."""
        ref_file = get_corresponding_reference_file_path(
            additional_identifier="beam", extension="vtu"
        )
        vtk_file = tmp_path / "test_beam.vtu"
        mesh.write_vtk(output_name="test", output_directory=tmp_path, binary=False)

        # Compare the vtk files.
        assert_results_close(ref_file, vtk_file)

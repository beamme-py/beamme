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

from beamme.core.element_beam import Beam3
from beamme.core.mesh import Mesh
from beamme.four_c.model_importer import import_four_c_model
from beamme.mesh_creation_functions.applications.beam_honeycomb import (
    create_beam_mesh_honeycomb,
)


def test_integration_core_mesh_visualization_beam(
    assert_results_close,
    get_default_test_beam_material,
    get_corresponding_reference_file_path,
):
    """Create a sample mesh and check the VTU output."""

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

    # Compare the vtu files.
    assert_results_close(
        get_corresponding_reference_file_path(extension="vtu"),
        mesh.get_vtu_representation(),
    )


def test_integration_core_mesh_visualization_solid(
    assert_results_close, get_corresponding_reference_file_path
):
    """Import a solid mesh and check the VTU output."""

    # Convert the solid mesh to beamme objects.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_tube"
        ),
        convert_input_to_mesh=True,
    )

    # Compare the vtu files.
    assert_results_close(
        get_corresponding_reference_file_path(extension="vtu"),
        mesh.get_vtu_representation(),
    )


def test_integration_core_mesh_visualization_solid_elements(
    assert_results_close, get_corresponding_reference_file_path
):
    """Import a solid mesh with all solid types and check the VTU output."""

    # Convert the solid mesh to beamme objects.
    _, mesh = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_multiple_solid_bricks"
        ),
        convert_input_to_mesh=True,
    )

    # Compare the vtu files.
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="solid", extension="vtu"
        ),
        mesh.get_vtu_representation(),
    )

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
"""Integration tests for the solid shell capability of BeamMe."""

import numpy as np

from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.four_c.model_importer import import_four_c_model
from beamme.four_c.solid_shell_thickness_direction import (
    get_visualization_third_parameter_direction_hex8,
    set_solid_shell_thickness_direction,
)
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_integration_four_c_solid_shell_thickness_direction_block(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test the solid shell direction detection functionality for a block geometry."""
    # Test the plates
    _, mesh_block = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_solid_shell_blocks"
        ),
        convert_input_to_mesh=True,
    )

    # Add a beam element to check the function also works with beam elements
    create_beam_mesh_line(
        mesh_block,
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        [0, 0, 0],
        [1, 0, 0],
        n_el=1,
    )
    # Set the thickness direction and compare result
    set_solid_shell_thickness_direction(mesh_block.elements, selection_type="thickness")
    assert_results_close(get_corresponding_reference_file_path(), mesh_block)


def test_integration_four_c_solid_shell_thickness_direction_dome(
    assert_results_close, get_corresponding_reference_file_path
):
    """Test the solid shell direction detection functionality for a dome geometry."""
    _, mesh_dome_original = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_solid_shell_dome"
        ),
        convert_input_to_mesh=True,
    )

    # Test that the thickness version works
    mesh_dome = mesh_dome_original.copy()
    set_solid_shell_thickness_direction(mesh_dome.elements, selection_type="thickness")
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="thickness"),
        mesh_dome,
    )

    # Test that the direction function version works (same result as thickness)
    def director_function(cell_center):
        """Return director that will be used to determine the solid thickness
        direction."""
        return cell_center / np.linalg.norm(cell_center)

    mesh_dome = mesh_dome_original.copy()
    set_solid_shell_thickness_direction(
        mesh_dome.elements,
        selection_type="projection_director_function",
        director_function=director_function,
    )
    assert_results_close(
        get_corresponding_reference_file_path(additional_identifier="thickness"),
        mesh_dome,
    )

    # Test that the constant direction version works
    mesh_dome = mesh_dome_original.copy()
    set_solid_shell_thickness_direction(
        mesh_dome.elements,
        selection_type="projection_director",
        director=[0, 0, 1],
        identify_threshold=None,
    )
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier="constant_direction"
        ),
        mesh_dome,
    )


def test_integration_four_c_solid_shell_thickness_direction_visualization(
    assert_results_close, get_corresponding_reference_file_path, tmp_path
):
    """Test the solid shell direction visualization functionality."""
    _, mesh_dome = import_four_c_model(
        input_file_path=get_corresponding_reference_file_path(
            reference_file_base_name="test_other_create_cubit_input_files_solid_shell_dome"
        ),
        convert_input_to_mesh=True,
    )

    ref_file = get_corresponding_reference_file_path(extension="vtu")
    test_file = tmp_path / ref_file.name

    set_solid_shell_thickness_direction(
        mesh_dome.elements,
        selection_type="projection_director",
        director=[0, 0, 1],
        identify_threshold=None,
    )
    grid = get_visualization_third_parameter_direction_hex8(mesh_dome)
    grid.save(test_file)
    assert_results_close(ref_file, test_file)

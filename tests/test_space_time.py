# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
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
"""Test the beam to space-time surface functionality."""

import numpy as np
import pytest

from beamme.core.conf import bme
from beamme.core.element_beam import generate_beam_class
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.mesh_creation_functions.beam_arc import create_beam_mesh_arc_segment_2d
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.space_time.beam_to_space_time import beam_to_space_time, mesh_to_data_arrays


def get_name(beam_class):
    """Return the identifier for the given beam object."""
    match len(beam_class.nodes_create):
        case 2:
            return "linear"
        case 3:
            return "quadratic"
        case _:
            raise TypeError("Got unexpected beam element")


@pytest.mark.parametrize("n_nodes", [2, 3])
def test_space_time_straight(
    n_nodes, assert_results_close, get_corresponding_reference_file_path
):
    """Create the straight beam for the tests."""

    # Create the beam mesh in space
    beam_type = generate_beam_class(n_nodes)
    beam_radius = 0.05
    mat = MaterialBeamBase(radius=beam_radius)
    mesh = Mesh()
    create_beam_mesh_line(mesh, beam_type, mat, [0, 0, 0], [6, 0, 0], n_el=3)

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 5, time_start=2.5)

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    additional_identifier = get_name(beam_type)
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.parametrize("n_nodes", [2, 3])
def test_space_time_curved(
    n_nodes, assert_results_close, get_corresponding_reference_file_path
):
    """Create a curved beam for the tests."""

    # Create the beam mesh in space
    beam_type = generate_beam_class(n_nodes)
    beam_radius = 0.05
    mat = MaterialBeamBase(radius=beam_radius)
    mesh = Mesh()
    create_beam_mesh_arc_segment_2d(
        mesh, beam_type, mat, [0.5, 1, 0], 0.75, 0.0, np.pi * 2.0 / 3.0, n_el=3
    )

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 5)

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    additional_identifier = get_name(beam_type)
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.parametrize("n_nodes", [2, 3])
@pytest.mark.parametrize("couple_nodes", [False, True])
def test_space_time_elbow(
    n_nodes, couple_nodes, assert_results_close, get_corresponding_reference_file_path
):
    """Create an elbow beam for the tests."""

    # Create the beam mesh in space
    beam_type = generate_beam_class(n_nodes)
    beam_radius = 0.05
    mat = MaterialBeamBase(radius=beam_radius)
    mesh = Mesh()
    create_beam_mesh_line(mesh, beam_type, mat, [0, 0, 0], [1, 0, 0], n_el=3)
    create_beam_mesh_line(mesh, beam_type, mat, [1, 0, 0], [1, 1, 0], n_el=2)

    # Create the couplings
    if couple_nodes:
        mesh.couple_nodes(
            reuse_matching_nodes=True, coupling_dof_type=bme.coupling_dof.fix
        )

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 5, time_start=1.69)

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    additional_identifier = get_name(beam_type) + ("_coupling" if couple_nodes else "")
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


@pytest.mark.parametrize("n_nodes", [2, 3])
@pytest.mark.parametrize("couple_nodes", [False, True])
@pytest.mark.parametrize("arc_length", [False, True])
def test_space_time_varying_material_length(
    n_nodes,
    couple_nodes,
    arc_length,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create an elbow beam for the tests."""

    beam_type = generate_beam_class(n_nodes)

    def beam_mesh_in_space_generator(time):
        """Create the beam mesh in space generator."""
        beam_radius = 0.05
        mat = MaterialBeamBase(youngs_modulus=100, radius=beam_radius, density=1)
        pos_y = 0.25 * (time - 1.7)

        mesh_1 = Mesh()
        create_beam_mesh_line(
            mesh_1,
            beam_type,
            mat,
            [np.sin(time), 0, 0],
            [2, pos_y, 0],
            n_el=3,
            set_nodal_arc_length=arc_length,
        )

        mesh_2 = Mesh()
        create_beam_mesh_line(
            mesh_2,
            beam_type,
            mat,
            [2, pos_y, 0],
            [2, 3, 0],
            n_el=2,
            set_nodal_arc_length=arc_length,
        )

        mesh = Mesh()
        mesh.add(mesh_1, mesh_2)
        if couple_nodes:
            mesh.couple_nodes(coupling_dof_type=bme.coupling_dof.fix)
        return mesh

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(
        beam_mesh_in_space_generator, 6.9, 5, time_start=1.69
    )

    # Add all sets to the mesh
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    additional_identifier = (
        get_name(beam_type)
        + ("_coupling" if couple_nodes else "")
        + ("_arc_length" if arc_length else "")
    )
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_close(
        get_corresponding_reference_file_path(
            additional_identifier=additional_identifier, extension="json"
        ),
        mesh_data_arrays,
    )


def test_space_time_named_node_set(
    assert_results_close, get_corresponding_reference_file_path
):
    """Create a straight beam and check that named node sets are handled
    correctly."""

    # Create the beam mesh in space
    mesh = Mesh()
    mat = MaterialBeamBase()
    beam_type = generate_beam_class(2)
    create_beam_mesh_line(mesh, beam_type, mat, [0, 0, 0], [6, 0, 0], n_el=2)

    # Get the space-time mesh
    space_time_mesh, return_set = beam_to_space_time(mesh, 6.9, 3, time_start=2.5)

    # Add all sets to the mesh
    return_set["start"].name = "start"
    return_set["right"].name = "right"
    return_set["surface"].name = "surface"
    space_time_mesh.add(return_set)

    # Check the mesh data arrays
    mesh_data_arrays = mesh_to_data_arrays(space_time_mesh)
    assert_results_close(
        get_corresponding_reference_file_path(extension="json"),
        mesh_data_arrays,
    )


@pytest.mark.performance
def test_performance_create_mesh_in_space(evaluate_execution_time, cache_data):
    """Test the performance of the mesh creation in space."""

    mesh = Mesh()
    beam_type = generate_beam_class(3)

    evaluate_execution_time(
        "BeamMe: Space-Time: Create mesh in space",
        create_beam_mesh_line,
        kwargs={
            "mesh": mesh,
            "beam_class": beam_type,
            "material": MaterialBeamBase(radius=0.05),
            "start_point": [0, 0, 0],
            "end_point": [1, 0, 0],
            "n_el": 100,
        },
        expected_time=0.01,
    )

    # store mesh in cache for upcoming test
    cache_data.mesh = mesh


@pytest.mark.performance
def test_performance_create_mesh_in_time(evaluate_execution_time, cache_data):
    """Test the performance of the mesh creation in time."""

    evaluate_execution_time(
        "BeamMe: Space-Time: Create mesh in time",
        beam_to_space_time,
        kwargs={
            "mesh_space_or_generator": cache_data.mesh,
            "time_duration": 6.9,
            "number_of_elements_in_time": 1000,
            "time_start": 1.69,
        },
        expected_time=3.0,
    )

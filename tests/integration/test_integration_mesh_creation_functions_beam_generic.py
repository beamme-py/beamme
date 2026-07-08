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
"""Integration tests for generic beam mesh creation functions."""

import autograd.numpy as npAD
import numpy as np
import pytest

from beamme.core.geometry_set import GeometryName, GeometrySet
from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_arc_segment_via_rotation,
)
from beamme.mesh_creation_functions.beam_helix import create_beam_mesh_helix
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.mesh_creation_functions.beam_parametric_curve import (
    create_beam_mesh_parametric_curve,
)

CLOSE_BEAM_ADDITIONAL_ROTATION_PARAMETRIZATION = (
    "additional_rotation",
    (None, Rotation([0, 1, 0], 0.5)),
)


@pytest.mark.parametrize(*CLOSE_BEAM_ADDITIONAL_ROTATION_PARAMETRIZATION)
def test_integration_mesh_creation_functions_beam_generic_close_beam_manual(
    additional_rotation,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circle mesh manually by creating the nodes and connecting them to the
    elements."""
    # Parameters for this test case.
    n_el = 3
    R = 1.235

    r = [R, 0, 0]
    if additional_rotation is None:
        basis = Rotation([0, 0, 1], np.pi * 0.5)
    else:
        basis = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)

    # Define material.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create the full circle manually.
    mesh = Mesh()
    mesh.add(mat)

    # Add nodes.
    for i in range(4 * n_el):
        node = NodeCosserat(r, basis)
        rotation = Rotation([0, 0, 1], 0.5 * i * np.pi / n_el)
        node.rotate(rotation, origin=[0, 0, 0])
        mesh.nodes.append(node)

    # Add elements.
    for i in range(2 * n_el):
        node_index = [2 * i, 2 * i + 1, 2 * i + 2]
        nodes = []
        for index in node_index:
            if index == len(mesh.nodes):
                nodes.append(mesh.nodes[0])
            else:
                nodes.append(mesh.nodes[index])
        element = Beam3rHerm2Line3(mat, nodes)
        mesh.add(element)

    # Add sets.
    geom_set = GeometryName()
    geom_set["start"] = GeometrySet(mesh.nodes[0])
    geom_set["end"] = GeometrySet(mesh.nodes[0])
    geom_set["line"] = GeometrySet(mesh.elements)
    mesh.add(geom_set)

    # Check the created mesh.
    additional_identifier = "rotation" if additional_rotation is not None else None
    assert_results_close(
        get_corresponding_reference_file_path(
            test_name_suffix_trim_count=1, additional_identifier=additional_identifier
        ),
        mesh,
    )


@pytest.mark.parametrize(*CLOSE_BEAM_ADDITIONAL_ROTATION_PARAMETRIZATION)
@pytest.mark.parametrize(
    "create_mesh_function",
    (create_beam_mesh_arc_segment_via_rotation, create_beam_mesh_parametric_curve),
)
def test_integration_mesh_creation_functions_beam_generic_close_beam_full_circle(
    additional_rotation,
    create_mesh_function,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a full circle with different mesh creation methods.

    Close the circle such that it is connected to its beginning.
    """
    # Parameters for this test case.
    R = 1.235
    n_el = 3

    # Define material.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create the mesh for different methods. The following code can be used for any
    # general mesh creation functions and can be done in a way where we either
    # create the full circle at once or create two half circles and connect them.
    mesh = Mesh()

    if additional_rotation is not None:
        start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
        start_node = NodeCosserat([R, 0, 0], start_rotation)
        mesh.add(start_node)
    else:
        start_node = None

    # Get the arguments for the mesh creation function.
    mesh_function_argument_dir = {
        "mesh": mesh,
        "beam_class": Beam3rHerm2Line3,
        "material": mat,
        "n_el": 2 * n_el,
        "close_beam": True,
        "start_node": start_node,
    }
    if create_mesh_function == create_beam_mesh_arc_segment_via_rotation:
        mesh_function_argument_dir["center"] = [0, 0, 0]
        mesh_function_argument_dir["axis_rotation"] = Rotation([0, 0, 1], np.pi / 2)
        mesh_function_argument_dir["radius"] = R
        mesh_function_argument_dir["angle"] = 2 * np.pi
    elif create_mesh_function == create_beam_mesh_parametric_curve:

        def circle_function(t):
            """Function for the circle."""
            return R * npAD.array([npAD.cos(t), npAD.sin(t)])

        mesh_function_argument_dir["function"] = circle_function
        mesh_function_argument_dir["interval"] = [0, 2 * np.pi]

    beam_sets = create_mesh_function(**mesh_function_argument_dir)
    mesh.add(beam_sets)

    # Check the created mesh.
    additional_identifier = "rotation" if additional_rotation is not None else None
    assert_results_close(
        get_corresponding_reference_file_path(
            test_name_suffix_trim_count=2, additional_identifier=additional_identifier
        ),
        mesh,
    )


@pytest.mark.parametrize(*CLOSE_BEAM_ADDITIONAL_ROTATION_PARAMETRIZATION)
@pytest.mark.parametrize(
    "create_mesh_function",
    (create_beam_mesh_arc_segment_via_rotation, create_beam_mesh_parametric_curve),
)
def test_integration_mesh_creation_functions_beam_generic_close_beam_two_circles(
    additional_rotation,
    create_mesh_function,
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a full circle with different mesh creation methods, each creates a half
    circle twice.

    Close the circle such that it is connected to its beginning.
    """
    # Parameters for this test case.
    R = 1.235
    n_el = 3

    # Define material.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create the mesh for different methods. The following code can be used for any
    # general mesh creation functions and can be done in a way where we either
    # create the full circle at once or create two half circles and connect them.
    mesh = Mesh()

    if additional_rotation is not None:
        start_rotation = additional_rotation * Rotation([0, 0, 1], np.pi * 0.5)
        start_node = NodeCosserat([R, 0, 0], start_rotation)
        mesh.add(start_node)
    else:
        start_node = None

    # Get the arguments for the mesh creation function.
    mesh_function_argument_dir_1 = {
        "mesh": mesh,
        "beam_class": Beam3rHerm2Line3,
        "material": mat,
        "n_el": n_el,
        "start_node": start_node,
    }
    if create_mesh_function == create_beam_mesh_arc_segment_via_rotation:
        mesh_function_argument_dir_1["center"] = [0, 0, 0]
        mesh_function_argument_dir_1["axis_rotation"] = Rotation([0, 0, 1], np.pi / 2)
        mesh_function_argument_dir_1["radius"] = R
        mesh_function_argument_dir_1["angle"] = np.pi

        mesh_function_argument_dir_2 = mesh_function_argument_dir_1.copy()
        mesh_function_argument_dir_2["axis_rotation"] = Rotation(
            [0, 0, 1], 3 * np.pi / 2
        )
    elif create_mesh_function == create_beam_mesh_parametric_curve:

        def circle_function(t):
            """Function for the circle."""
            return R * npAD.array([npAD.cos(t), npAD.sin(t)])

        mesh_function_argument_dir_1["function"] = circle_function
        mesh_function_argument_dir_1["interval"] = [0, np.pi]

        mesh_function_argument_dir_2 = mesh_function_argument_dir_1.copy()
        mesh_function_argument_dir_2["interval"] = [np.pi, 2 * np.pi]

    beam_sets_1 = create_mesh_function(**mesh_function_argument_dir_1)

    mesh_function_argument_dir_2["start_node"] = beam_sets_1["end"]
    mesh_function_argument_dir_2["end_node"] = beam_sets_1["start"]
    beam_sets_2 = create_mesh_function(**mesh_function_argument_dir_2)

    geom_set = GeometryName()
    geom_set["start"] = GeometrySet(beam_sets_1["start"])
    geom_set["end"] = GeometrySet(beam_sets_2["end"])
    geom_set["line"] = GeometrySet([beam_sets_1["line"], beam_sets_2["line"]])
    mesh.add(geom_set)

    # Check the created mesh.
    additional_identifier = "rotation" if additional_rotation is not None else None
    assert_results_close(
        get_corresponding_reference_file_path(
            test_name_suffix_trim_count=2, additional_identifier=additional_identifier
        ),
        mesh,
    )


def test_integration_mesh_creation_functions_beam_generic_node_positions_of_elements_option(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Creates a line, a circular segments in 2D and a helix by setting the
    node_positions_of_elements."""
    # Create a mesh
    mesh = Mesh()

    # Create and add material to mesh.
    material = get_default_test_beam_material(material_type="reissner")
    mesh.add(material)

    # Create a beam line with specified node_positions_of_elements.
    create_beam_mesh_line(
        mesh,
        Beam3rHerm2Line3,
        material,
        [-1, -1, 0],
        [-1, -1, 3],
        node_positions_of_elements=[0, 0.1, 0.5, 0.9, 1.0],
    )

    # Create an arc segment similar to the equally spaced one,
    # but based on the provided node_positions_of_elements.
    create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        material,
        [1.0, 2.0, 0.0],
        1.5,
        np.pi * 0.25,
        np.pi * (1.0 + 1.0 / 3.0),
        node_positions_of_elements=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    # Create a 2d segments with different element sizes.
    create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        material,
        [1.0, 2.0, 0.0],
        2.0,
        np.pi * 0.25,
        np.pi * (1.0 + 1.0 / 3.0),
        node_positions_of_elements=[0, 1.0 / 3.0, 1.0],
    )

    # Create a helix with different node positions.
    create_beam_mesh_helix(
        mesh,
        Beam3rHerm2Line3,
        material,
        [0.0, 0.0, 1.0],
        [4.0, 4.0, 0.0],
        [5.0, 5.0, 0.0],
        height_helix=10.0,
        turns=2.5 / np.pi,
        node_positions_of_elements=[0, 1.0 / 3.0, 1],
    )

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_generic_element_length_option(
    get_default_test_beam_material,
    get_parametric_function,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that the element length can be specified in the beam creation functions."""
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")

    l_el = 1.5

    mesh_line = Mesh()
    create_beam_mesh_line(
        mesh_line,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 6.0],
        l_el=l_el,
    )

    mesh_line_long = Mesh()
    create_beam_mesh_line(
        mesh_line_long,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 2.0],
        [3.0, 4.0, 8.0],
        l_el=100,
    )

    mesh_arc = Mesh()
    create_beam_mesh_arc_segment_via_rotation(
        mesh_arc,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 3.0],
        Rotation([1, 3, 4], np.pi / 3.0),
        2.0,
        np.pi * 2.0 / 3.0,
        l_el=l_el,
    )

    # Get the helix curve function
    R = 2.0
    tz = 4.0  # incline
    n = 0.5  # number of turns
    helix = get_parametric_function(
        "helix", R, tz, transformation_factor=2.0, number_of_turns=n
    )

    mesh_curve = Mesh()
    create_beam_mesh_parametric_curve(
        mesh_curve,
        Beam3rHerm2Line3,
        mat,
        helix,
        [0.0, 2.0 * np.pi * n],
        l_el=l_el,
    )

    # Check the output
    mesh.add(mesh_line, mesh_line_long, mesh_arc, mesh_curve)
    assert_results_close(get_corresponding_reference_file_path(), mesh)

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
"""This script is used to test the arc mesh creation functions."""

import numpy as np
import pytest

from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_arc import (
    create_beam_mesh_arc_segment_2d,
    create_beam_mesh_arc_segment_via_axis,
    create_beam_mesh_arc_segment_via_rotation,
)


def test_integration_mesh_creation_functions_beam_arc_segment_via_axis(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circular segment via the axis method and compare it with the reference
    file."""
    # Create mesh
    mesh = Mesh()
    radius = 2.0
    beam_set = create_beam_mesh_arc_segment_via_axis(
        mesh,
        Beam3rHerm2Line3,
        get_default_test_beam_material(material_type="reissner"),
        [0, 0, 1],
        [0, radius, 0],
        [0, 0, 0],
        1.0,
        n_el=3,
    )
    mesh.add(beam_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_arc_segment_start_end_node(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Check that if start end nodes with non-matching positions or tangents are
    provided we get an error.

    TODO: Split this test into the arc functionality and a general unittest for
    the error messages.
    """
    angle = 1.0
    radius = 2.0
    start_node_pos = [0, 0, 0]
    start_node_rot = Rotation()
    end_node_pos = radius * np.array([np.sin(angle), 1.0 - np.cos(angle), 0])
    end_node_rot = Rotation([0, 0, 1], angle)

    def create_beam(*, start_node=None, end_node=None):
        """This is the base function we use to generate the beam in this test case."""
        mesh = Mesh()
        mat = get_default_test_beam_material(material_type="reissner")
        if start_node is not None:
            mesh.add(start_node)
        if end_node is not None:
            mesh.add(end_node)
        create_beam_mesh_arc_segment_via_axis(
            mesh,
            Beam3rHerm2Line3,
            mat,
            [0, 0, 1],
            [0, radius, 0],
            [0, 0, 0],
            angle,
            n_el=3,
            start_node=start_node,
            end_node=end_node,
        )
        return mesh

    # This should work as expected, as all the values match.
    start_node = NodeCosserat(start_node_pos, start_node_rot)
    end_node = NodeCosserat(end_node_pos, end_node_rot)
    mesh = create_beam(start_node=start_node, end_node=end_node)
    assert_results_close(get_corresponding_reference_file_path(), mesh)

    # Create with start node where the position does not match.
    with pytest.raises(
        ValueError,
        match="The position of the given start node does not match with the position from the function!",
    ):
        start_node = NodeCosserat([0, 1, 0], start_node_rot)
        create_beam(start_node=start_node)

    # Create with start node where the rotation does not match.
    with pytest.raises(
        ValueError,
        match="The tangent of the start node does not match with the given function!",
    ):
        start_node = NodeCosserat(start_node_pos, Rotation([1, 2, 3], np.pi / 3))
        create_beam(start_node=start_node)

    # Create with end node where the position does not match.
    with pytest.raises(
        ValueError,
        match="The position of the given end node does not match with the position from the function!",
    ):
        end_node = NodeCosserat([0, 0, 0], end_node_rot)
        create_beam(end_node=end_node)

    # Create with end node where the rotation does not match.
    with pytest.raises(
        ValueError,
        match="The tangent of the end node does not match with the given function!",
    ):
        end_node = NodeCosserat(end_node_pos, Rotation([1, 2, 3], np.pi / 3))
        create_beam(end_node=end_node)


def test_integration_mesh_creation_functions_beam_arc_segment_via_rotation(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circular segment via the rotation method and compare it with the
    reference file."""
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create mesh.
    beam_set = create_beam_mesh_arc_segment_via_rotation(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [3, 6, 9.2],
        Rotation([4.5, 7, 10], np.pi / 5),
        10,
        np.pi / 2.3,
        n_el=5,
    )

    # Add boundary conditions.
    mesh.add(beam_set)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)


def test_integration_mesh_creation_functions_beam_arc_segment_2d(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Create a circular segments in 2D."""
    # Create mesh
    mesh = Mesh()

    # Add material and function.
    mat = get_default_test_beam_material(material_type="reissner")

    # Create mesh.
    beam_set_1 = create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 0.0],
        1.5,
        np.pi * 0.25,
        np.pi * (1.0 + 1.0 / 3.0),
        n_el=5,
    )
    beam_set_2 = create_beam_mesh_arc_segment_2d(
        mesh,
        Beam3rHerm2Line3,
        mat,
        [1.0, 2.0, 0.0] - 2.0 * 0.5 * np.array([1, np.sqrt(3), 0]),
        0.5,
        np.pi / 3.0,
        -np.pi,
        n_el=3,
        start_node=mesh.nodes[-1],
    )

    # Add geometry sets
    mesh.add(beam_set_1, beam_set_2)

    # Check the output.
    assert_results_close(get_corresponding_reference_file_path(), mesh)

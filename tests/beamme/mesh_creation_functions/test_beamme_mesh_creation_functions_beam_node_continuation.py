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
"""Unit tests for the node continuation beam mesh creation functions."""

import numpy as np

from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_node_continuation import (
    create_beam_mesh_arc_at_node,
)


def test_beamme_mesh_creation_functions_beam_node_continuation_accumulated_arc(
    get_default_test_beam_material, assert_results_close
):
    """Test that the arc node continuation function can be applied multiple times in a
    row.

    This function can lead to accumulated errors in the rotations if not implemented
    carefully.
    """
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    n_segments = 100

    rotation_ref = Rotation([1, 0, 0], 0.5 * np.pi) * Rotation([0, 0, 1], 0.5 * np.pi)
    start_node = NodeCosserat([1, 2, 3], rotation_ref)
    mesh.add(start_node)
    beam_set = {"end": start_node}
    angle = np.pi
    angle_increment = angle / n_segments
    axis = [1, 0, 0]
    for i in range(n_segments):
        beam_set = create_beam_mesh_arc_at_node(
            mesh,
            Beam3rHerm2Line3,
            mat,
            beam_set["end"],
            axis,
            1.0,
            angle_increment,
            n_el=2,
        )

    rotation_actual = beam_set["end"].get_points()[0].rotation

    # Calculate the solution and get the "analytical" solution
    rotation_expected = Rotation(axis, angle) * rotation_ref
    quaternion_expected = np.array([-0.5, 0.5, -0.5, -0.5])
    assert rotation_actual == rotation_expected
    assert_results_close(rotation_actual.q, quaternion_expected)

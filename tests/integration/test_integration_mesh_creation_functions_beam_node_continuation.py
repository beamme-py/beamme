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
"""Integration tests for node continuation mesh creation functions."""

import numpy as np

from beamme.core.mesh import Mesh
from beamme.core.node import NodeCosserat
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.mesh_creation_functions.beam_node_continuation import (
    create_beam_mesh_arc_at_node,
    create_beam_mesh_line_at_node,
)


def test_integration_mesh_creation_functions_beam_node_continuation_line_and_arc(
    get_default_test_beam_material,
    assert_results_close,
    get_corresponding_reference_file_path,
):
    """Test that the node continuation function work as expected."""
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="reissner")
    mesh.add(mat)

    start_node = NodeCosserat([1, 2, 3], Rotation())
    mesh.add(start_node)
    beam_set = create_beam_mesh_line_at_node(
        mesh, Beam3rHerm2Line3, mat, start_node, 1.2, n_el=1
    )
    beam_set = create_beam_mesh_arc_at_node(
        mesh,
        Beam3rHerm2Line3,
        mat,
        beam_set["end"],
        [0, 1, 0],
        1.0,
        np.pi * 0.5,
        n_el=1,
    )
    beam_set = create_beam_mesh_arc_at_node(
        mesh,
        Beam3rHerm2Line3,
        mat,
        beam_set["end"],
        [0, 1, 0],
        1.0,
        -np.pi * 0.5,
        n_el=2,
    )
    beam_set = create_beam_mesh_arc_at_node(
        mesh,
        Beam3rHerm2Line3,
        mat,
        beam_set["end"],
        [0, 0, 1],
        1.0,
        -np.pi * 3.0 / 4.0,
        n_el=1,
    )
    create_beam_mesh_line_at_node(
        mesh, Beam3rHerm2Line3, mat, beam_set["end"], 2.3, n_el=3
    )

    assert_results_close(get_corresponding_reference_file_path(), mesh)

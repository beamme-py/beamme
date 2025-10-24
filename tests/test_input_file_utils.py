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
"""This script is used to test the functionality of the input file utils."""

from beamme.core.mesh import Mesh
from beamme.core.mesh_utils import get_coupled_nodes_to_master_map
from beamme.four_c.element_beam import Beam3rLine2Line2
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_input_file_utils_get_coupled_nodes_to_master_map(
    get_default_test_beam_material,
):
    """Test the get_coupled_nodes_to_master_map function."""

    beam_class = Beam3rLine2Line2
    mat = get_default_test_beam_material(material_type="reissner")
    mesh = Mesh()
    create_beam_mesh_line(mesh, beam_class, mat, [0, 0, 0], [1, 0, 0])
    create_beam_mesh_line(mesh, beam_class, mat, [1, 0, 0], [1, 1, 0])
    create_beam_mesh_line(mesh, beam_class, mat, [1, 1, 0], [2, 0, 0])
    create_beam_mesh_line(mesh, beam_class, mat, [1, 0, 0], [2, 0, 0])

    mesh.couple_nodes()

    replaced_node_to_master_map, unique_nodes = get_coupled_nodes_to_master_map(
        mesh, assign_i_global=True
    )

    assert len(unique_nodes) == 4
    assert unique_nodes == [mesh.nodes[i_node] for i_node in [0, 1, 3, 5]]

    assert len(replaced_node_to_master_map) == 4
    assert replaced_node_to_master_map[mesh.nodes[2]] == mesh.nodes[1]
    assert replaced_node_to_master_map[mesh.nodes[4]] == mesh.nodes[3]
    assert replaced_node_to_master_map[mesh.nodes[7]] == mesh.nodes[5]
    assert replaced_node_to_master_map[mesh.nodes[6]] == mesh.nodes[1]

    expected_i_global = [0, 1, 1, 2, 2, 3, 1, 3]
    for i_node, expected_index in enumerate(expected_i_global):
        assert mesh.nodes[i_node].i_global == expected_index

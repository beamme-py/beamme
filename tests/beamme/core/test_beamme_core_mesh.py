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
"""This script is used to test the functionality of the core mesh."""

import numpy as np
import pytest

from beamme.core.conf import bme
from beamme.core.coupling import Coupling
from beamme.core.element_beam import Beam, Beam3
from beamme.core.geometry_set import GeometrySet
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.core.node import Node
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


def test_beamme_core_mesh_get_nodes_by_function(assert_results_close):
    """Check if the get_nodes_by_function method of Mesh works properly."""

    def get_nodes_at_x(node, x_value):
        """True for all coordinates at a certain x value."""
        if np.abs(node.coordinates[0] - x_value) < 1e-10:
            return True
        else:
            return False

    mat = MaterialBeamBase()

    mesh = Mesh()
    create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [5, 0, 0], n_el=5)
    create_beam_mesh_line(mesh, Beam3, mat, [0, 1, 0], [10, 1, 0], n_el=10)

    nodes = mesh.get_nodes_by_function(get_nodes_at_x, 1.0)
    assert 2 == len(nodes)
    for node in nodes:
        assert_results_close(1.0, node.coordinates[0])


def test_beamme_core_mesh_add_checks():
    """This test checks that Mesh raises an error when double objects are added to the
    mesh."""
    # Mesh instance for this test.
    mesh = Mesh()

    # Create basic objects that will be added to the mesh.
    node = Node([0, 1.0, 2.0])
    element = Beam()
    mesh.add(node)
    mesh.add(element)

    # Create objects based on basic mesh items.
    coupling = Coupling(mesh.nodes, bme.bc.point_coupling, bme.coupling_dof.fix)
    coupling_penalty = Coupling(
        mesh.nodes, bme.bc.point_coupling_penalty, bme.coupling_dof.fix
    )
    geometry_set = GeometrySet(mesh.elements)
    mesh.add(coupling)
    mesh.add(coupling_penalty)
    mesh.add(geometry_set)

    # Add the objects again and check for errors.
    with pytest.raises(ValueError, match="The node is already in this mesh!"):
        mesh.add(node)
    with pytest.raises(ValueError, match="The element is already in this mesh!"):
        mesh.add(element)
    with pytest.raises(ValueError, match="The item is already in this container!"):
        mesh.add(coupling)
    with pytest.raises(ValueError, match="The item is already in this container!"):
        mesh.add(coupling_penalty)
    with pytest.raises(ValueError, match="The item is already in this container!"):
        mesh.add(geometry_set)


def test_beamme_core_mesh_multiple_couple_nodes():
    """The current implementation can handle more than one coupling on a node correctly,
    therefore we check this here."""
    # Create mesh object
    mesh = Mesh()
    mat = MaterialBeamBase()

    # Add two beams to create an elbow structure. The beams each have a
    # node at the intersection
    beam_set_1 = create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [1, 0, 0])
    beam_set_2 = create_beam_mesh_line(mesh, Beam3, mat, [1, 0, 0], [1, 1, 0])

    # Call coupling twice -> this will create two coupling objects for the
    # corner node
    mesh.couple_nodes()
    mesh.couple_nodes()

    # Check that we have two point couplings and that they both couple the end and
    # start node of the created lines.
    couplings = mesh.boundary_conditions[bme.bc.point_coupling, bme.geo.point]
    assert len(couplings) == 2
    for coupling in couplings:
        coupling_nodes = coupling.geometry_set.get_points()
        assert len(coupling_nodes) == 2
        assert beam_set_1["end"].get_points()[0] in coupling_nodes
        assert beam_set_2["start"].get_points()[0] in coupling_nodes


def test_beamme_core_mesh_get_named_geometry_sets():
    """Test the `get_named_geometry_sets` method."""
    mesh = Mesh()
    mat = MaterialBeamBase()
    beam_set_1 = create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [1, 0, 0])
    beam_set_2 = create_beam_mesh_line(mesh, Beam3, mat, [1, 0, 0], [1, 1, 0])

    # Name some of the geometry sets
    beam_set_1["start"].name = "beam_1_start"
    beam_set_2["start"].name = "beam_2_start"
    beam_set_2["line"].name = "beam_2_line"

    mesh.add(beam_set_1, beam_set_2)

    # Check that the correct geometry sets are returned
    named_geometry_sets = mesh.get_named_geometry_sets()
    assert len(named_geometry_sets) == 3
    assert (
        set(["beam_1_start", "beam_2_start", "beam_2_line"])
        == named_geometry_sets.keys()
    )
    for name, node_indices in zip(
        ["beam_1_start", "beam_2_start", "beam_2_line"], [[0], [3], [3, 4, 5]]
    ):
        nodes = named_geometry_sets[name].get_all_nodes()
        assert len(node_indices) == len(nodes)
        for node_index, node in zip(node_indices, nodes):
            assert mesh.nodes[node_index] is node


def test_beamme_core_mesh_get_named_geometry_sets_error():
    """Test that the `get_named_geometry_sets` method raises an error when geometry sets
    have duplicate names."""
    mesh = Mesh()
    mat = MaterialBeamBase()
    beam_set = create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [1, 0, 0])

    # Name two geometry sets with the same name
    beam_set["start"].name = "node_set"
    beam_set["end"].name = "node_set"

    mesh.add(beam_set)

    with pytest.raises(ValueError, match=r"Geometry set name node_set is not unique."):
        mesh.get_named_geometry_sets()

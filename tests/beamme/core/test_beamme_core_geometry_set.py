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
"""This script is used to unittest the functionality of the geometry sets."""

from typing import Callable

import pytest

from beamme.core.conf import bme
from beamme.core.element_beam import Beam, Beam3
from beamme.core.geometry_set import GeometrySet, GeometrySetNodes
from beamme.core.mesh import Mesh
from beamme.core.node import Node, NodeCosserat
from beamme.core.rotation import Rotation
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line


@pytest.fixture()
def assert_geometry_set_add_operator() -> Callable:
    """Return a function to check the results in the geometry set operator tests."""

    def _compare_results(
        mesh_objects, combined_geometry, set_1_geometry, set_2_geometry
    ):
        """Compare the results."""
        # Check that the added geometry set contains the combined geometry
        assert len(combined_geometry) == 5
        assert combined_geometry[0] is mesh_objects[2]
        assert combined_geometry[1] is mesh_objects[3]
        assert combined_geometry[2] is mesh_objects[4]
        assert combined_geometry[3] is mesh_objects[0]
        assert combined_geometry[4] is mesh_objects[1]

        # Check that the original sets are not modified
        assert len(set_1_geometry) == 3
        assert set_1_geometry[0] is mesh_objects[0]
        assert set_1_geometry[1] is mesh_objects[1]
        assert set_1_geometry[2] is mesh_objects[2]
        assert len(set_2_geometry) == 3
        assert set_2_geometry[0] is mesh_objects[2]
        assert set_2_geometry[1] is mesh_objects[3]
        assert set_2_geometry[2] is mesh_objects[4]

    return _compare_results


@pytest.mark.parametrize(
    ("mesh_object", "mesh_object_args"),
    [
        (Node, [[1, 2, 3]]),
        (Beam, []),
    ],
)
def test_beamme_core_geometry_set_add_operator(
    mesh_object, mesh_object_args, assert_geometry_set_add_operator
):
    """Test that geometry sets can be added to each other.

    We test this once with a point geometry set based on nodes and once with a line
    geometry set based on beam elements.
    """
    mesh_objects = [mesh_object(*mesh_object_args) for _ in range(5)]
    set_1 = GeometrySet(mesh_objects[:3])
    set_2 = GeometrySet(mesh_objects[2:])
    combined_set = set_2 + set_1
    combined_geometry = combined_set.get_geometry_objects()

    assert_geometry_set_add_operator(
        mesh_objects,
        combined_geometry,
        set_1.get_geometry_objects(),
        set_2.get_geometry_objects(),
    )


@pytest.mark.parametrize("geometry_type", [bme.geo.point, bme.geo.line])
def test_beamme_core_geometry_set_nodes_add_operator(
    geometry_type, assert_geometry_set_add_operator
):
    """Test that node based geometry sets can be added to each other."""
    mesh_objects = [Node([1, 2, 3]) for _ in range(5)]
    set_1 = GeometrySetNodes(geometry_type, nodes=mesh_objects[:3])
    set_2 = GeometrySetNodes(geometry_type, nodes=mesh_objects[2:])
    combined_set = set_2 + set_1
    combined_geometry = combined_set.get_all_nodes()

    assert_geometry_set_add_operator(
        mesh_objects, combined_geometry, set_1.get_all_nodes(), set_2.get_all_nodes()
    )


def test_beamme_core_geometry_set_add():
    """Test functionality of the GeometrySet add method."""
    mesh = Mesh()
    for i in range(6):
        mesh.add(NodeCosserat([i, 2 * i, 3 * i], Rotation()))

    set_1 = GeometrySetNodes(
        bme.geo.point, [mesh.nodes[0], mesh.nodes[1], mesh.nodes[2]]
    )
    set_2 = GeometrySetNodes(
        bme.geo.point, [mesh.nodes[2], mesh.nodes[3], mesh.nodes[4]]
    )
    set_12 = GeometrySetNodes(bme.geo.point)
    set_12.add(set_1)
    set_12.add(set_2)
    set_3 = GeometrySet(set_1.get_points())

    mesh.add(set_1, set_2, set_12, set_3)

    # Check the resulting sets
    unique_sets = mesh.get_unique_geometry_sets()
    for key, value in unique_sets.items():
        if key is bme.geo.point:
            assert len(value) == 4
        else:
            assert len(value) == 0

    results = [
        {"len": 3, "indices": [0, 1, 2]},
        {"len": 3, "indices": [2, 3, 4]},
        {"len": 5, "indices": [0, 1, 2, 3, 4]},
        {"len": 3, "indices": [0, 1, 2]},
    ]
    for i_set, result_dict in enumerate(results):
        point_set = unique_sets[bme.geo.point][i_set]
        nodes = point_set.get_all_nodes()
        assert len(nodes) == result_dict["len"]
        for i_node, node_index in enumerate(result_dict["indices"]):
            assert nodes[i_node] is mesh.nodes[node_index]


def test_beamme_core_geometry_set_unique_ordering_of_get_all_nodes_for_line_condition(
    get_default_test_beam_material,
):
    """This test ensures that the ordering of the nodes returned from the function
    get_all_nodes is unique for line sets."""
    # set up a beam mesh with material
    mesh = Mesh()
    mat = get_default_test_beam_material(material_type="base")
    beam_set = create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [2, 0, 0], n_el=10)

    # check the nodes in the line set
    nodes_set = beam_set["line"].get_all_nodes()
    assert len(nodes_set) == 21
    mesh_node_indices = range(21)
    for i_node_set, i_node_mesh in enumerate(mesh_node_indices):
        assert nodes_set[i_node_set] is mesh.nodes[i_node_mesh]


def test_beamme_core_geometry_set_get_geometry_objects(get_default_test_beam_material):
    """Test if the geometry set returns the objects(elements) in the correct order."""
    # Initialize material and mesh
    mat = get_default_test_beam_material(material_type="base")
    mesh = Mesh()

    # number of elements
    n_el = 5

    # Create a simple beam.
    geometry = create_beam_mesh_line(mesh, Beam3, mat, [0, 0, 0], [2, 0, 0], n_el=n_el)

    # Get all elements from the geometry set.
    elements_of_geometry = geometry["line"].get_geometry_objects()

    # Check number of elements.
    assert len(elements_of_geometry) == n_el

    # Check if the order of the elements from the geometry set is the same as for the mesh.
    for i_element, element in enumerate(elements_of_geometry):
        assert element == mesh.elements[i_element]

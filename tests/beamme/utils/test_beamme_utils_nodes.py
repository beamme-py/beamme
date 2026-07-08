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
"""This script is used to unit test functions for nodes."""

import numpy as np

from beamme.core.node import Node, NodeCosserat
from beamme.core.rotation import Rotation
from beamme.utils.nodes import (
    adjust_close_nodes,
    get_min_max_coordinates,
    is_node_on_plane,
)


def test_beamme_utils_nodes_adjusting_of_nodes(assert_results_close):
    """Test the mesh function adjust_close_nodes."""
    coordinates = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1.3, 0, 0], [1.0, 0.3, 0], [2, 0, 0.2]]
    )
    coordinates_averaged = np.array(
        [
            [0, 0, 0],
            [1.1, 0.1, 0],
            [2, 0, 0.1],
            [1.1, 0.1, 0],
            [1.1, 0.1, 0],
            [2, 0, 0.1],
        ]
    )
    nodes = [Node(coordinates=coord) for coord in coordinates]
    adjust_close_nodes(nodes, tol=0.35)
    assert_results_close(coordinates_averaged, [node.coordinates for node in nodes])


def test_beamme_utils_nodes_is_node_on_plane():
    """Test if node on plane function works properly."""
    # node on plane with origin_distance
    node = Node([1.0, 1.0, 1.0])
    assert is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=1.0)

    # node on plane with point_on_plane
    node = Node([1.0, 1.0, 1.0])
    assert is_node_on_plane(
        node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
    )

    # node not on plane with origin_distance
    node = Node([13.5, 14.5, 15.5])
    assert not is_node_on_plane(node, normal=[0.0, 0.0, 1.0], origin_distance=5.0)

    # node not on plane with point_on_plane
    node = Node([13.5, 14.5, 15.5])
    assert not is_node_on_plane(
        node, normal=[0.0, 0.0, 5.0], point_on_plane=[5.0, 5.0, 1.0]
    )


def test_beamme_utils_nodes_get_min_max_coordinates(assert_results_close):
    """Test if the get_min_max_coordinates function works properly."""
    # Create the mesh.
    nodes = []
    nodes.append(Node([0.0, 0.0, 0.0]))
    nodes.append(Node([0.0, 0.0, -1.5]))
    nodes.append(Node([-0.5, 1.0, 3.0]))
    nodes.append(Node([0.5, -1.0, 4.0]))
    nodes.append(NodeCosserat([0.0, 0.0, 0.0], Rotation()))
    nodes.append(NodeCosserat([-0.5, 3.0, 4.0], Rotation()))
    nodes.append(NodeCosserat([2.0, -1.0, 0.0], Rotation()))

    # Check the results.
    min_max = get_min_max_coordinates(nodes)
    ref_solution = [-0.5, -1.0, -1.5, 2.0, 3.0, 4.0]
    assert_results_close(min_max, ref_solution)

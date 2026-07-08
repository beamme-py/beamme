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
"""Helper functions to find, filter and interact with nodes."""

import numpy as _np
from numpy.typing import NDArray as _NDArray

from beamme.core.conf import bme as _bme
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.node import Node as _Node
from beamme.core.node import NodeCosserat as _NodeCosserat
from beamme.geometric_search.find_close_points import (
    find_close_points as _find_close_points,
)
from beamme.geometric_search.find_close_points import (
    point_partners_to_partner_indices as _point_partners_to_partner_indices,
)


def find_close_nodes(nodes: list[_Node], **kwargs) -> list[list[_Node]]:
    """Find nodes in a point cloud that are within a certain tolerance of each other.

    Args:
        nodes: Nodes who are part of the point cloud.
        **kwargs: Arguments passed on to geometric_search.find_close_points

    Returns:
        A list of lists of nodes that are close to each other, i.e.,
        each element in the returned list contains nodes that are close
        to each other.
    """
    coords = _np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coords[i, :] = node.coordinates
    partner_indices = _point_partners_to_partner_indices(
        *_find_close_points(coords, **kwargs)
    )
    return [[nodes[i] for i in partners] for partners in partner_indices]


def adjust_close_nodes(nodes: list[_Node], *, tol=_bme.eps_pos) -> None:
    """Adjust the coordinates of nodes that are within the given tolerance by setting
    all involved coordinates of the nodes to their common mean.

    Args:
        nodes: List of nodes whose coordinates need adjustment.
        tol: Distance tolerance used to detect partner nodes.
    """
    partner_nodes = find_close_nodes(nodes, tol=tol)
    for close_nodes in partner_nodes:
        average_coords = _np.mean([node.coordinates for node in close_nodes], axis=0)
        for node in close_nodes:
            node.coordinates = average_coords.copy()


def check_node_by_coordinate(node, axis, value, eps=_bme.eps_pos):
    """Check if the node is at a certain coordinate value.

    Args
    ----
    node: Node
        The node to be checked for its position.
    axis: int
        Coordinate axis to check.
        0 -> x, 1 -> y, 2 -> z
    value: float
        Value for the coordinate that the node should have.
    eps: float
        Tolerance to check for equality.
    """
    return _np.abs(node.coordinates[axis] - value) < eps


def get_min_max_coordinates(nodes):
    """Return an array with the minimal and maximal coordinates of the given nodes.

    Return
    ----
    min_max_coordinates:
        [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    coordinates = _np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coordinates[i, :] = node.coordinates
    min_max = _np.zeros(6)
    min_max[:3] = _np.min(coordinates, axis=0)
    min_max[3:] = _np.max(coordinates, axis=0)
    return min_max


def get_single_node(item: _Node | _GeometrySetBase) -> _NodeCosserat:
    """Function to get a single node from the input item.

    Args:
        item: This can be a GeometrySet with exactly one node or a single node object.

    Returns:
        If a single node, or a Geometry set (point set) containing a single node
        is given, that node is returned, otherwise an error is raised.
    """
    if isinstance(item, _Node):
        node = item
    elif isinstance(item, _GeometrySetBase):
        # Check if there is only one node in the set
        nodes = item.get_points()
        if len(nodes) == 1:
            node = nodes[0]
        else:
            raise ValueError("GeometrySet does not have exactly one node!")
    else:
        raise TypeError(
            f'The given object can be node or GeometrySet got "{type(item)}"!'
        )

    if not isinstance(node, _NodeCosserat):
        raise TypeError("Expected a NodeCosserat object.")

    return node


def filter_nodes(nodes, *, middle_nodes=True) -> list[_Node]:
    """Filter the list of the given nodes.

    Be aware that if no filters are enabled the original list will be returned.

    Args
    ----
    nodes: list(Nodes)
        If this list is given it will be returned as is.
    middle_nodes: bool
        If middle nodes should be returned or not.
    """
    if not middle_nodes:
        return [node for node in nodes if middle_nodes or not node.is_middle_node]
    else:
        return nodes


def get_nodal_coordinates(nodes: list[_Node]) -> _NDArray:
    """Return an array with the coordinates of the given nodes.

    Args:
        nodes: Nodes for which the coordinates should be returned.

    Returns:
        Numpy array with all the positions of the nodes.
    """
    coordinates = _np.zeros([len(nodes), 3])
    for i, node in enumerate(nodes):
        coordinates[i, :] = node.coordinates
    return coordinates


def get_nodal_quaternions(nodes: list[_Node]) -> _NDArray:
    """Return an array with the quaternions of the given nodes.

    Args:
        nodes: List of nodes where we want the quaternion array.
    Returns:
        A numpy array containing the quaternions (the length is the number of
        nodes and the dtype is a numpy quaternion). For nodes which don't
        contain a rotation, we set the dummy quaternion (2, 0, 0, 0).
    """
    quaternions = _np.zeros([len(nodes), 4])
    for i, node in enumerate(nodes):
        if isinstance(node, _NodeCosserat):
            quaternions[i, :] = node.rotation.get_quaternion()
        else:
            # For the case of nodes that belong to solid elements,
            # we define the following default value:
            quaternions[i, :] = [2.0, 0.0, 0.0, 0.0]
    return quaternions


def get_nodes_by_function(nodes, function, *args, middle_nodes=False, **kwargs):
    """Return all nodes for which the function evaluates to true.

    Args
    ----
    nodes: [Node]
        Nodes that should be filtered.
    function: function(node, *args, **kwargs)
        Nodes for which this function is true are returned.
    middle_nodes: bool
        If this is true, middle nodes of a beam are also returned.
    """
    node_list = filter_nodes(nodes, middle_nodes=middle_nodes)
    return [node for node in node_list if function(node, *args, **kwargs)]


def get_min_max_nodes(nodes, *, middle_nodes=False):
    """Return a geometry set with the max and min nodes in all directions.

    Args
    ----
    nodes: list(Nodes)
        If this one is given return an array with the coordinates of the
        nodes in list, otherwise of all nodes in the mesh.
    middle_nodes: bool
        If this is true, middle nodes of a beam are also returned.
    """
    node_list = filter_nodes(nodes, middle_nodes=middle_nodes)
    geometry = _GeometryName()

    pos = get_nodal_coordinates(node_list)
    for i, direction in enumerate(["x", "y", "z"]):
        # Check if there is more than one value in dimension.
        min_max = [_np.min(pos[:, i]), _np.max(pos[:, i])]
        if _np.abs(min_max[1] - min_max[0]) >= _bme.eps_pos:
            for j, text in enumerate(["min", "max"]):
                # get all nodes with the min / max coordinate
                min_max_nodes = []
                for index, value in enumerate(
                    _np.abs(pos[:, i] - min_max[j]) < _bme.eps_pos
                ):
                    if value:
                        min_max_nodes.append(node_list[index])
                geometry[f"{direction}_{text}"] = _GeometrySet(min_max_nodes)
    return geometry


def is_node_on_plane(
    node, *, normal=None, origin_distance=None, point_on_plane=None, tol=_bme.eps_pos
):
    """Query if a node lies on a plane defined by a point_on_plane or the origin
    distance.

    Args
    ----
    node:
        Check if this node coincides with the defined plane.
    normal: _np.array, list
        Normal vector of defined plane.
    origin_distance: float
        Distance between origin and defined plane. Mutually exclusive with
        point_on_plane.
    point_on_plane: _np.array, list
        Point on defined plane. Mutually exclusive with origin_distance.
    tol: float
        Tolerance of evaluation if point coincides with plane

    Return
    ----
    True if the point lies on the plane, False otherwise.
    """
    if origin_distance is None and point_on_plane is None:
        raise ValueError("Either provide origin_distance or point_on_plane!")
    elif origin_distance is not None and point_on_plane is not None:
        raise ValueError("Only provide origin_distance OR point_on_plane!")

    if origin_distance is not None:
        projection = _np.dot(node.coordinates, normal) / _np.linalg.norm(normal)
        distance = _np.abs(projection - origin_distance)
    elif point_on_plane is not None:
        distance = _np.abs(
            _np.dot(point_on_plane - node.coordinates, normal) / _np.linalg.norm(normal)
        )

    return distance < tol

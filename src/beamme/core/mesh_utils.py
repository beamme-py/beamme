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
"""This module defines utility functions for meshes."""

from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple

import numpy as _np

from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.core.mesh_representation import (
    string_to_geometry_set_info as _string_to_geometry_set_info,
)
from beamme.core.node import Node as _Node


def get_coupled_nodes_to_master_map(
    mesh: _Mesh, *, assign_i_global: bool = False
) -> _Tuple[_Dict[_Node, _Node], _List[_Node]]:
    """Get a mapping of nodes in a mesh that should be "replaced" because they are
    coupled via a joint.

    In some finite element (FE) solvers, nodes coupled via joints are resolved
    by assigning a "master" node to represent the joint. This function identifies
    such nodes and creates a mapping where each coupled node is mapped to its
    master node.

    Args
    ----
    mesh:
        Input mesh
    assign_i_global:
        If this flag is set, the global indices are set in the node objects.

    Return
    ----
    replaced_node_to_master_map:
        A dictionary mapping each "replaced" node to its "master" node.
    unique_nodes:
        A list containing all unique nodes in the mesh, i.e., all nodes which
        are not coupled and the master nodes.
    """
    # Get a dictionary that maps the "replaced" nodes to the "master" ones
    replaced_node_to_master_map = {}
    for coupling in mesh.boundary_conditions[_bme.bc.point_coupling, _bme.geo.point]:
        if coupling.data is not _bme.coupling_dof.fix:
            raise ValueError(
                "This function is only implemented for rigid joints at the DOFs"
            )
        coupling_nodes = coupling.geometry_set.get_points()
        for node in coupling_nodes[1:]:
            replaced_node_to_master_map[node] = coupling_nodes[0]

    # Check that no "replaced" node is a "master" node
    master_nodes = set(replaced_node_to_master_map.values())
    for replaced_node in replaced_node_to_master_map.keys():
        if replaced_node in master_nodes:
            raise ValueError(
                "A replaced node is also a master nodes. This is not supported"
            )

    # Get all unique nodes
    unique_nodes = [
        node for node in mesh.nodes if node not in replaced_node_to_master_map
    ]

    # Optionally number the nodes
    if assign_i_global:
        for i_node, node in enumerate(unique_nodes):
            node.i_global = i_node
        for replaced_node, master_node in replaced_node_to_master_map.items():
            replaced_node.i_global = master_node.i_global

    # Return the mapping
    return replaced_node_to_master_map, unique_nodes


def apply_nodal_coupling_to_mesh_representation(
    mesh_representation: _MeshRepresentation,
    geometry_sets_to_i_global: _Dict[_GeometrySetBase, int],
    coupling_conditions: list[_Coupling],
):
    """Modify a mesh representation such that coupled nodes are represented by a single
    node.

    Args:
        mesh_representation: The mesh representation where coupling nodes should
            be merged. Will be modified in-place.
        geometry_sets_to_i_global: A mapping from geometry sets to their global
            IDs.
        coupling_conditions: A list of coupling conditions that define which
            nodes should be coupled.
    """
    # Get a dictionary that maps the "replaced" nodes to the "master" ones
    replaced_node_to_master_map = {}
    for coupling in coupling_conditions:
        if not coupling.data == _bme.coupling_dof.fix:
            raise ValueError(
                "This function is only implemented for rigid joints at the DOFs"
            )
        geometry_set_i_global = geometry_sets_to_i_global[coupling.geometry_set]
        for name in mesh_representation.point_data.keys():
            info = _string_to_geometry_set_info(name)
            if info is not None and info.i_global == geometry_set_i_global:
                coupled_node_ids = mesh_representation.point_data[name].nonzero()[0]
                for node in coupled_node_ids[1:]:
                    replaced_node_to_master_map[node] = coupled_node_ids[0]
                break
        else:
            raise ValueError(
                f"Could not find geometry set with i_global {geometry_set_i_global} in the mesh representation"
            )

    # Check that no "replaced" node is a "master" node
    master_nodes = set(replaced_node_to_master_map.values())
    replaced_nodes = set(replaced_node_to_master_map.keys())
    if len(master_nodes & replaced_nodes) > 0:
        raise ValueError(
            "A replaced node is also a master nodes. This is not supported. "
            f"Replaced nodes: {replaced_nodes}, master nodes: {master_nodes}, "
            f"intersection: {master_nodes & replaced_nodes}"
        )

    # Get mask that filters out replaced nodes
    point_mask = _np.ones(mesh_representation.n_points, dtype=bool)
    point_mask[list(replaced_nodes)] = False

    # Get the node mapping vector from the old IDs to the new ones
    mapping_vector = _np.zeros(mesh_representation.n_points, dtype=int)
    mapping_vector[point_mask] = _np.arange(
        mesh_representation.n_points - len(replaced_nodes)
    )
    for replaced_node, master_node in replaced_node_to_master_map.items():
        # Map the replaced node to the (new) ID of the master node
        mapping_vector[replaced_node] = mapping_vector[master_node]

    # Merge geometry-set membership of replaced nodes into their master nodes before
    # removing points (otherwise geometry sets / BC references can be lost).
    for name, values in mesh_representation.point_data.items():
        if _string_to_geometry_set_info(name) is not None:
            for replaced_node, master_node in replaced_node_to_master_map.items():
                values[master_node] = max(values[master_node], values[replaced_node])

    # Filter the point data vectors in the mesh representation.
    # For now, we simply drop the "replaced" points.
    mesh_representation.points = mesh_representation.points[point_mask]
    for name in mesh_representation.point_data.keys():
        mesh_representation.point_data[name] = mesh_representation.point_data[name][
            point_mask
        ]

    # Adapt the connectivity such that the "replaced" nodes are mapped to the "master" ones.
    connectivity_mask = _np.ones(mesh_representation.cell_connectivity.size, dtype=bool)
    connectivity_mask[mesh_representation.cell_connectivity_offsets[:-1]] = False
    point_indices_old = mesh_representation.cell_connectivity[connectivity_mask]
    point_indices_new = mapping_vector[point_indices_old]
    mesh_representation.cell_connectivity[connectivity_mask] = point_indices_new

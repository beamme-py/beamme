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
"""Convert a beam to a space time surface mesh."""

from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union
from typing import cast as _cast

import numpy as _np
import vtk as _vtk

from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.element_volume import VolumeElement as _VolumeElement
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_utils import (
    get_coupled_nodes_to_master_map as _get_coupled_nodes_to_master_map,
)
from beamme.core.node import NodeCosserat as _NodeCosserat
from beamme.utils.nodes import get_nodal_coordinates as _get_nodal_coordinates


class NodeCosseratSpaceTime(_NodeCosserat):
    """A Cosserat node in space-time.

    We add the 4th dimension time as a class variable.
    """

    def __init__(self, coordinates, rotation, time, **kwargs):
        super().__init__(coordinates, rotation, **kwargs)
        self.time = time


class SpaceTimeElement(_VolumeElement):
    """A general beam space-time surface element."""

    def __init__(self, nodes, **kwargs):
        super().__init__(nodes=nodes, **kwargs)


class SpaceTimeElementQuad4(SpaceTimeElement):
    """A space-time element with 4 nodes."""

    vtk_cell_type = _vtk.vtkQuad
    vtk_topology = list(range(4))


class SpaceTimeElementQuad9(SpaceTimeElement):
    """A space-time element with 9 nodes."""

    vtk_cell_type = _vtk.vtkQuadraticQuad
    vtk_topology = list(range(9))


def beam_to_space_time(
    mesh_space_or_generator: _Union[_Mesh, _Callable[[float], _Mesh]],
    time_duration: float,
    number_of_elements_in_time: int,
    *,
    time_start: float = 0.0,
) -> _Tuple[_Mesh, _GeometryName]:
    """Convert a beam mesh to a surface space-time mesh.

    Args:
        mesh_space_or_generator:
            Either a fixed spatial Mesh object or a function that returns the
            spatial mesh for a given time. If this is a generator, the topology
            of the mesh at the initial time is chosen for all times, only the
            positions and rotations are updated.
        time_duration:
            Total time increment to be solved with the space-time mesh
        number_of_elements_in_time:
            Number of elements in time direction
        time_start:
            Starting time for the space-time mesh. Can be used to create time slaps.
    Returns:
        Tuple (space_time_mesh, return_set)
        - space_time_mesh:
            The space time mesh. Be aware that translating / rotating this mesh
            might lead to unexpected results.
        - return_set:
            The nodes sets to be returned for the space time mesh:
                "start", "end", "left", "right", "surface"
    """

    # Get the "reference" spatial mesh
    if callable(mesh_space_or_generator):
        mesh_space_reference = mesh_space_or_generator(time_start)
    else:
        mesh_space_reference = mesh_space_or_generator

    # Perform some sanity checks
    element_types = {type(element) for element in mesh_space_reference.elements}
    if not len(element_types) == 1:
        raise ValueError(
            f"Expected all elements to be of the same type, got {element_types}"
        )
    element_type = element_types.pop()

    # Calculate global mesh properties
    number_of_nodes_in_space = len(mesh_space_reference.nodes)
    number_of_elements_in_space = len(mesh_space_reference.elements)
    space_time_element_type: _Union[
        _Type[SpaceTimeElementQuad4], _Type[SpaceTimeElementQuad9]
    ]
    if len(element_type.nodes_create) == 2:
        number_of_copies_in_time = number_of_elements_in_time + 1
        time_increment_between_nodes = time_duration / number_of_elements_in_time
        space_time_element_type = SpaceTimeElementQuad4
    elif len(element_type.nodes_create) == 3:
        number_of_copies_in_time = 2 * number_of_elements_in_time + 1
        time_increment_between_nodes = time_duration / (2 * number_of_elements_in_time)
        space_time_element_type = SpaceTimeElementQuad9
    else:
        raise TypeError(f"Got unexpected element type {element_type}")

    # Number the nodes in the original mesh
    for i_node, node in enumerate(mesh_space_reference.nodes):
        node.i_global = i_node

    # Get the nodes for the final space-time mesh
    left_nodes = []
    right_nodes = []
    space_time_nodes = []
    for i_mesh_space in range(number_of_copies_in_time):
        time = time_increment_between_nodes * i_mesh_space + time_start

        if callable(mesh_space_or_generator):
            mesh_space_current_time = mesh_space_or_generator(time)
            if (not len(mesh_space_current_time.nodes) == number_of_nodes_in_space) or (
                not len(mesh_space_current_time.elements) == number_of_elements_in_space
            ):
                raise ValueError(
                    "The number of nodes and elements does not match for the generated "
                    "space time meshes."
                )
        else:
            mesh_space_current_time = mesh_space_reference

        space_time_nodes_to_add = [
            NodeCosseratSpaceTime(
                node.coordinates, node.rotation, time, arc_length=node.arc_length
            )
            for node in mesh_space_current_time.nodes
        ]
        space_time_nodes.extend(space_time_nodes_to_add)

        if i_mesh_space == 0:
            start_nodes = space_time_nodes_to_add
        elif i_mesh_space == number_of_copies_in_time - 1:
            end_nodes = space_time_nodes_to_add

        left_nodes.append(space_time_nodes_to_add[0])
        right_nodes.append(space_time_nodes_to_add[-1])

    # Create the space time elements
    space_time_elements = []
    for i_element_time in range(number_of_elements_in_time):
        for element in mesh_space_reference.elements:
            element_node_ids = [node.i_global for node in element.nodes]
            if space_time_element_type == SpaceTimeElementQuad4:
                # Create the indices for the linear element
                first_time_row_start_index = i_element_time * number_of_nodes_in_space
                second_time_row_start_index = (
                    1 + i_element_time
                ) * number_of_nodes_in_space
                element_node_indices = [
                    first_time_row_start_index + element_node_ids[0],
                    first_time_row_start_index + element_node_ids[1],
                    second_time_row_start_index + element_node_ids[1],
                    second_time_row_start_index + element_node_ids[0],
                ]
            elif space_time_element_type == SpaceTimeElementQuad9:
                # Create the indices for the quadratic element
                first_time_row_start_index = (
                    2 * i_element_time * number_of_nodes_in_space
                )
                second_time_row_start_index = (
                    2 * i_element_time + 1
                ) * number_of_nodes_in_space
                third_time_row_start_index = (
                    2 * i_element_time + 2
                ) * number_of_nodes_in_space
                element_node_indices = [
                    first_time_row_start_index + element_node_ids[0],
                    first_time_row_start_index + element_node_ids[2],
                    third_time_row_start_index + element_node_ids[2],
                    third_time_row_start_index + element_node_ids[0],
                    first_time_row_start_index + element_node_ids[1],
                    second_time_row_start_index + element_node_ids[2],
                    third_time_row_start_index + element_node_ids[1],
                    second_time_row_start_index + element_node_ids[0],
                    second_time_row_start_index + element_node_ids[1],
                ]
            else:
                raise TypeError(
                    f"Got unexpected space time element type {space_time_element_type}"
                )

            # Add the element to the mesh
            space_time_elements.append(
                space_time_element_type(
                    [space_time_nodes[i_node] for i_node in element_node_indices]
                )
            )

    # Add joints to the space time mesh
    space_time_couplings = []
    for coupling in mesh_space_reference.boundary_conditions[
        _bme.bc.point_coupling, _bme.geo.point
    ]:
        for i_mesh_space in range(number_of_copies_in_time):
            coupling_node_ids = [
                node.i_global for node in coupling.geometry_set.get_points()
            ]
            space_time_couplings.append(
                _Coupling(
                    [
                        space_time_nodes[node_id + number_of_nodes_in_space]
                        for node_id in coupling_node_ids
                    ],
                    coupling.bc_type,
                    coupling.data,
                )
            )

    # Create the new mesh and add all the mesh items
    space_time_mesh = _Mesh()
    space_time_mesh.add(space_time_nodes)
    space_time_mesh.add(space_time_elements)
    space_time_mesh.add(space_time_couplings)

    # Create the element sets
    return_set = _GeometryName()
    return_set["start"] = _GeometrySet(start_nodes)
    return_set["end"] = _GeometrySet(end_nodes)
    return_set["left"] = _GeometrySet(left_nodes)
    return_set["right"] = _GeometrySet(right_nodes)
    return_set["surface"] = _GeometrySetNodes(_bme.geo.surface, space_time_mesh.nodes)

    return space_time_mesh, return_set


def mesh_to_data_arrays(mesh: _Mesh):
    """Get the relevant data arrays from the space time mesh."""

    element_types = list(set([type(element) for element in mesh.elements]))
    if len(element_types) > 1:
        raise ValueError("Got more than a single element type, this is not supported")
    elif not (
        element_types[0] == SpaceTimeElementQuad4
        or element_types[0] == SpaceTimeElementQuad9
    ):
        raise TypeError(
            f"Expected either SpaceTimeElementQuad4 or SpaceTimeElementQuad9, got {element_types[0]}"
        )

    _, raw_unique_nodes = _get_coupled_nodes_to_master_map(mesh, assign_i_global=True)
    unique_nodes = _cast(_List[NodeCosseratSpaceTime], raw_unique_nodes)

    n_nodes = len(unique_nodes)
    n_elements = len(mesh.elements)
    n_nodes_per_element = len(mesh.elements[0].nodes)

    coordinates = _get_nodal_coordinates(unique_nodes)
    time = _np.zeros(n_nodes)
    connectivity = _np.zeros((n_elements, n_nodes_per_element), dtype=int)
    element_rotation_vectors = _np.zeros((n_elements, n_nodes_per_element, 3))

    for i_node, node in enumerate(unique_nodes):
        time[i_node] = node.time

    for i_element, element in enumerate(mesh.elements):
        for i_node, node in enumerate(element.nodes):
            connectivity[i_element, i_node] = node.i_global
            element_rotation_vectors[i_element, i_node, :] = (
                node.rotation.get_rotation_vector()
            )

    geometry_sets = mesh.get_unique_geometry_sets()
    node_sets: _Dict[str, _Dict] = {}
    for value in geometry_sets.values():
        for geometry_set in value:
            node_set_data = {
                "node_ids": _np.array(
                    [node.i_global for node in geometry_set.get_all_nodes()]
                )
            }
            if geometry_set.name is not None:
                node_set_data["name"] = geometry_set.name
            node_sets[str(len(node_sets) + 1)] = node_set_data

    return_dict = {
        "coordinates": coordinates,
        "time": time,
        "connectivity": connectivity,
        "element_rotation_vectors": element_rotation_vectors,
        "node_sets": node_sets,
    }

    nodes_have_arc_length = {node.arc_length is not None for node in mesh.nodes}
    if len(nodes_have_arc_length) > 1:
        raise ValueError(
            "Some nodes have an arc length, some don't. This is not supported."
        )
    if nodes_have_arc_length.pop():
        # The arc length is added as an "element" property, since the same
        # node can have a different arc length depending on the element
        # (similar to the rotation vectors).
        arc_length = _np.zeros((n_elements, n_nodes_per_element))
        for i_element, element in enumerate(mesh.elements):
            for i_node, node in enumerate(element.nodes):
                connectivity[i_element, i_node] = node.i_global
                arc_length[i_element, i_node] = node.arc_length
        return_dict["arc_length"] = arc_length

    return return_dict

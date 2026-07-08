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
"""Convert a beam to a space time surface mesh."""

from typing import Callable as _Callable
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import cast as _cast

import numpy as _np
import pyvista as _pv

from beamme.core.conf import bme as _bme
from beamme.core.coupling import Coupling as _Coupling
from beamme.core.element_volume import VolumeElement as _VolumeElement
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.geometry_set import GeometrySetNodes as _GeometrySetNodes
from beamme.core.mesh import Mesh as _Mesh
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.core.mesh_utils import (
    apply_nodal_coupling_to_mesh_representation as _apply_nodal_coupling_to_mesh_representation,
)
from beamme.core.node import Node as _Node
from beamme.core.node import NodeCosserat as _NodeCosserat


class NodeCosseratSpaceTime(_NodeCosserat):
    """A Cosserat node in space-time.

    We add the 4th dimension time as a class variable.
    """

    node_type = _bme.node_type.space_time_cosserat

    def __init__(self, coordinates, rotation, time, **kwargs):
        super().__init__(coordinates, rotation, **kwargs)
        self.time = time


class SpaceTimeElement(_VolumeElement):
    """A general beam space-time surface element."""

    def __init__(self, nodes, **kwargs):
        super().__init__(nodes=nodes, **kwargs)


class SpaceTimeElementQuad4(SpaceTimeElement):
    """A space-time element with 4 nodes."""

    element_type = _bme.element_type.space_time_beam
    vtk_cell_type = _pv.CellType.QUAD
    data = {}


class SpaceTimeElementQuad9(SpaceTimeElement):
    """A space-time element with 9 nodes."""

    element_type = _bme.element_type.space_time_beam
    vtk_cell_type = _pv.CellType.BIQUADRATIC_QUAD
    data = {}


def beam_to_space_time(
    mesh_space_or_generator: _Mesh | _Callable[[float], _Mesh],
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
                "start", "end", "surface"
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
    space_time_element_type: _Type[SpaceTimeElementQuad4] | _Type[SpaceTimeElementQuad9]

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

    # Number nodes and elements in the original mesh
    for i_node, node in enumerate(mesh_space_reference.nodes):
        node.i_global = i_node
    for i_element, element in enumerate(mesh_space_reference.elements):
        element.i_global = i_element

    # Get the nodes for the final space-time mesh
    space_time_nodes = []
    start_nodes: list[_Node] = []
    end_nodes: list[_Node] = []
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
            start_nodes.extend(space_time_nodes_to_add)
        elif i_mesh_space == number_of_copies_in_time - 1:
            end_nodes.extend(space_time_nodes_to_add)

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
    coupling_geometry_sets = set()
    for coupling in mesh_space_reference.boundary_conditions[
        _bme.bc.point_coupling, _bme.geo.point
    ]:
        coupling_set = coupling.geometry_set
        coupling_geometry_sets.add(coupling_set)
        coupling_node_ids = [node.i_global for node in coupling_set.get_points()]
        for i_mesh_space in range(number_of_copies_in_time):
            space_time_couplings.append(
                _Coupling(
                    [
                        space_time_nodes[
                            node_id + i_mesh_space * number_of_nodes_in_space
                        ]
                        for node_id in coupling_node_ids
                    ],
                    coupling.bc_type,
                    coupling.data,
                )
            )

    # Convert geometry sets to the space time mesh
    raise_geometry_type = {
        _bme.geo.point: _bme.geo.line,
        _bme.geo.line: _bme.geo.surface,
        _bme.geo.surface: _bme.geo.volume,
    }
    all_sets_in_space = mesh_space_reference.get_unique_geometry_sets()
    space_time_geometry_sets: list[_GeometrySetBase] = []
    for geometry_type, geometry_sets in all_sets_in_space.items():
        for geometry_set in geometry_sets:
            if geometry_set in coupling_geometry_sets:
                # The coupling geometry sets are already handled above, so we skip them here.
                continue

            if isinstance(geometry_set, _GeometrySet) and (
                geometry_type == _bme.geo.line
                or geometry_type == _bme.geo.surface
                or geometry_type == _bme.geo.volume
            ):
                raised_geometry_set_elements = []
                for element in geometry_set.get_geometry_objects():
                    for i_element_row_in_time in range(number_of_elements_in_time):
                        raised_geometry_set_elements.append(
                            space_time_elements[
                                _cast(int, element.i_global)
                                + i_element_row_in_time * number_of_elements_in_space
                            ]
                        )
                space_time_geometry_sets.append(
                    _GeometrySet(
                        raised_geometry_set_elements,
                        name=geometry_set.name,
                    )
                )

            else:
                geometry_set_nodes = geometry_set.get_all_nodes()
                raised_geometry_set_nodes = []
                for node in geometry_set_nodes:
                    for i_mesh_space in range(number_of_copies_in_time):
                        raised_geometry_set_nodes.append(
                            space_time_nodes[
                                node.i_global + i_mesh_space * number_of_nodes_in_space
                            ]
                        )

                geometry_type_raised = raise_geometry_type[geometry_type]
                space_time_geometry_sets.append(
                    _GeometrySetNodes(
                        geometry_type_raised,
                        raised_geometry_set_nodes,
                        name=geometry_set.name,
                    )
                )

    # Create the new mesh and add all the mesh items
    space_time_mesh = _Mesh()
    space_time_mesh.add(space_time_nodes)
    space_time_mesh.add(space_time_elements)
    space_time_mesh.add(space_time_couplings)
    space_time_mesh.add(space_time_geometry_sets)

    # Create the element sets
    return_set = _GeometryName()
    return_set["start"] = _GeometrySetNodes(_bme.geo.line, start_nodes)
    return_set["end"] = _GeometrySetNodes(_bme.geo.line, end_nodes)
    return_set["surface"] = _GeometrySetNodes(_bme.geo.surface, space_time_mesh.nodes)

    return space_time_mesh, return_set


def get_space_time_mesh_representation(mesh: _Mesh) -> _MeshRepresentation:
    """Get the mesh representation for the space time mesh.

    Compared to the standard mesh representation, coupled nodes are represented by the
    same node. This requires some additional element data arrays which are added by
    this function.

    Args:
        mesh: The space time mesh.

    Returns:
        The mesh representation for the space time mesh.
    """
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

    # Number of nodes per element
    n_nodes_per_element = len(mesh.elements[0].nodes)

    # Get the mesh representation
    (mesh_representation, _, geometry_sets_to_i_global, _) = (
        mesh.get_mesh_representation()
    )

    # Get the element rotation vectors and arc length values. This has to be done before
    # the coupled nodes are removed.
    point_rotation_vectors = mesh_representation.point_data["rotation_vector"]
    element_rotation_vectors = _np.zeros(
        (mesh_representation.n_cells, n_nodes_per_element * 3)
    )
    element_arc_lengths = None
    if "arc_length" in mesh_representation.point_data:
        point_arc_lengths = mesh_representation.point_data["arc_length"]
        element_arc_lengths = _np.zeros(
            (mesh_representation.n_cells, n_nodes_per_element)
        )

    for i_element, connectivity in enumerate(
        mesh_representation.connectivity_iterator()
    ):
        for i_local, i_global in enumerate(connectivity):
            element_rotation_vectors[i_element, i_local * 3 : (i_local + 1) * 3] = (
                point_rotation_vectors[i_global]
            )
            if element_arc_lengths is not None:
                element_arc_lengths[i_element, i_local] = point_arc_lengths[i_global]

    # Add the element data arrays
    mesh_representation.cell_data["rotation_vector"] = element_rotation_vectors
    if element_arc_lengths is not None:
        mesh_representation.cell_data["arc_length"] = element_arc_lengths

    # Apply the coupling by explicitly replacing the coupled nodes.
    _apply_nodal_coupling_to_mesh_representation(
        mesh_representation,
        geometry_sets_to_i_global,
        mesh.boundary_conditions[_bme.bc.point_coupling, _bme.geo.point],
    )

    return mesh_representation

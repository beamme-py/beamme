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
"""This module defines the Mesh class, which holds the content (nodes,
elements, sets, ...) for a meshed geometry."""

import copy as _copy
import warnings as _warnings
from pathlib import Path as _Path
from typing import Any as _Any
from typing import List as _List
from typing import cast as _cast

import numpy as _np
import pyvista as _pv
import quaternion as _quaternion
from numpy.typing import NDArray as _NDArray

from beamme.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from beamme.core.boundary_condition import (
    BoundaryConditionContainer as _BoundaryConditionContainer,
)
from beamme.core.conf import bme as _bme
from beamme.core.coupling import coupling_factory as _coupling_factory
from beamme.core.element import Element as _Element
from beamme.core.element_beam import Beam as _Beam
from beamme.core.function import Function as _Function
from beamme.core.geometry_set import GeometryName as _GeometryName
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.geometry_set import GeometrySetContainer as _GeometrySetContainer
from beamme.core.material import Material as _Material
from beamme.core.mesh_representation import (
    MESH_REPRESENTATION_MAPPINGS as _MESH_REPRESENTATION_MAPPINGS,
)
from beamme.core.mesh_representation import GeometrySetInfo as _GeometrySetInfo
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.core.mesh_representation import (
    string_to_geometry_set_info as _string_to_geometry_set_info,
)
from beamme.core.node import Node as _Node
from beamme.core.node import NodeCosserat as _NodeCosserat
from beamme.core.nurbs_patch import NURBSPatch as _NURBSPatch
from beamme.core.rotation import Rotation as _Rotation
from beamme.core.rotation import add_rotations as _add_rotations
from beamme.core.rotation import rotate_coordinates as _rotate_coordinates
from beamme.geometric_search.find_close_points import (
    find_close_points as _find_close_points,
)
from beamme.utils.environment import is_testing as _is_testing
from beamme.utils.nodes import filter_nodes as _filter_nodes
from beamme.utils.nodes import find_close_nodes as _find_close_nodes
from beamme.utils.nodes import get_nodal_coordinates as _get_nodal_coordinates
from beamme.utils.nodes import get_nodal_quaternions as _get_nodal_quaternions
from beamme.utils.nodes import get_nodes_by_function as _get_nodes_by_function


class Mesh:
    """A class that contains a full mesh, i.e. Nodes, Elements, Boundary
    Conditions, Sets, Couplings, Materials and Functions."""

    def __init__(self):
        """Initialize all empty containers."""

        self.nodes = []
        self.elements = []
        self.materials = []
        self.functions = []
        self.geometry_sets = _GeometrySetContainer()
        self.boundary_conditions = _BoundaryConditionContainer()

    @staticmethod
    def get_base_mesh_item_type(item):
        """Return the base mesh type of the given item.

        Amongst other things, we need this function so we can check if
        all items of a list are of the same "base" type.

        Args:
            item: The object we want to get the base type from.
        """

        for cls in (
            Mesh,
            _Function,
            _BoundaryConditionBase,
            _Material,
            _Node,
            _Element,
            _GeometrySetBase,
            _GeometryName,
        ):
            if isinstance(item, cls):
                return cls
        return type(item)

    def add(self, *args, **kwargs):
        """Add an item to this mesh, depending on its type.

        If an list is given each list element is added with this
        function. If multiple arguments are given, each one is
        individually added with this function. Keyword arguments are
        passed through to the adding function.
        """

        match len(args):
            case 0:
                raise ValueError("At least one argument is required!")
            case 1:
                add_item = args[0]
                base_type = self.get_base_mesh_item_type(add_item)

                base_type_to_method_map = {
                    Mesh: self.add_mesh,
                    _Function: self.add_function,
                    _BoundaryConditionBase: self.add_bc,
                    _Material: self.add_material,
                    _Node: self.add_node,
                    _Element: self.add_element,
                    _GeometrySetBase: self.add_geometry_set,
                    _GeometryName: self.add_geometry_name,
                    list: self.add_list,
                }
                if base_type in base_type_to_method_map:
                    base_type_to_method_map[base_type](add_item, **kwargs)
                else:
                    raise TypeError(
                        f'No Mesh.add case implemented for type: "{type(add_item)}" with base type "{base_type}"!'
                    )
            case _:
                for item in args:
                    self.add(item, **kwargs)

    def add_mesh(self, mesh):
        """Add the content of another mesh to this mesh."""

        # Add each item from mesh to self.
        self.add(mesh.nodes)
        self.add(mesh.elements)
        self.add(mesh.materials)
        self.add(mesh.functions)
        self.geometry_sets.extend(mesh.geometry_sets)
        self.boundary_conditions.extend(mesh.boundary_conditions)

    def add_bc(self, bc):
        """Add a boundary condition to this mesh."""
        bc_key = bc.bc_type
        geom_key = bc.geometry_set.geometry_type
        bc.geometry_set.check_replaced_nodes()
        self.boundary_conditions.append((bc_key, geom_key), bc)

    def add_function(self, function):
        """Add a function to this mesh item.

        Check that the function is only added once.
        """
        if function not in self.functions:
            self.functions.append(function)

    def add_material(self, material):
        """Add a material to this mesh item.

        Check that the material is only added once.
        """
        if material not in self.materials:
            self.materials.append(material)

    def add_node(self, node):
        """Add a node to this mesh."""
        if node in self.nodes:
            raise ValueError("The node is already in this mesh!")
        self.nodes.append(node)

    def add_element(self, element):
        """Add an element to this mesh."""
        if element in self.elements:
            raise ValueError("The element is already in this mesh!")
        self.elements.append(element)

    def add_geometry_set(self, geometry_set):
        """Add a geometry set to this mesh."""
        geometry_set.check_replaced_nodes()
        self.geometry_sets.append(geometry_set.geometry_type, geometry_set)

    def add_geometry_name(self, geometry_name):
        """Add a set of geometry sets to this mesh.

        Sort by the keys here to create a deterministic ordering,
        especially for testing purposes
        """
        keys = list(geometry_name.keys())
        keys.sort()
        for key in keys:
            self.add(geometry_name[key])

    def add_list(self, add_list: _List, **kwargs) -> None:
        """Add a list of items to this mesh.

        Args:
            add_list:
                List to be added to the mesh. This method checks that all
                base types of the list items are the same.

                In the special case of a node or element list, we add the whole list
                at once. This avoids a check for duplicate entries for every
                addition, which scales very badly. By doing it this way we only
                check the final list for duplicate entries which is much more
                performant.

                For all other types of items, we add each element individually
                via the Mesh.add method.
        """

        types = {self.get_base_mesh_item_type(item) for item in add_list}
        if len(types) > 1:
            raise TypeError(
                f"You can only add lists with the same type of element. Got {types}"
            )
        elif len(types) == 1:
            list_type = types.pop()

            def extend_internal_list(self_list: _List, new_list: _List) -> None:
                """Extend an internal list with the new list.

                It is checked that the final list does not have
                duplicate entries.
                """
                self_list.extend(new_list)
                if not len(set(self_list)) == len(self_list):
                    raise ValueError(
                        "The added list contains entries already existing in the Mesh"
                    )

            if list_type == _Node:
                extend_internal_list(self.nodes, add_list)
            elif list_type == _Element:
                extend_internal_list(self.elements, add_list)
            else:
                for item in add_list:
                    self.add(item, **kwargs)

    def replace_nodes(self, replace_nodes: dict[_Node, _Node]) -> None:
        """Replace all nodes in this mesh that are in the replacement map.

        Args:
            replace_nodes: A dictionary that maps source nodes to target nodes. The
                source nodes will be replaced with the target nodes in the mesh.
        """

        # Nothing to do if the replacement map is empty.
        if len(replace_nodes) == 0:
            return

        # Check that all source and target nodes are in the mesh.
        for items, name in (
            (replace_nodes.keys(), "source"),
            (replace_nodes.values(), "target"),
        ):
            mesh_contains_all_nodes = set(items).issubset(self.nodes)
            if not mesh_contains_all_nodes:
                raise ValueError(f"Not all {name} nodes are in the mesh!")

        # Remove source nodes from the mesh.
        self.nodes = [node for node in self.nodes if node not in replace_nodes]

        # Replace the nodes in the elements.
        for element in self.elements:
            for i, node in enumerate(element.nodes):
                if node in replace_nodes:
                    element.nodes[i] = replace_nodes[node]

        # Set the links to the target nodes in the source nodes, so they can be
        # replaced in the geometry sets.
        for source_node, target_node in replace_nodes.items():
            source_node.target_node = target_node

        # Replace the nodes in the geometry sets.
        mesh_sets = self.get_unique_geometry_sets()
        for geometry_set_list in mesh_sets.values():
            for geometry_set in geometry_set_list:
                geometry_set.check_replaced_nodes()

    def get_unique_geometry_sets(
        self, *, coupling_sets: bool = True
    ) -> _GeometrySetContainer:
        """Return a geometry set container that contains geometry sets
        explicitly added to the mesh, as well as sets for boundary conditions.

        The i_global values are set in the returned geometry sets.

        Args:
            coupling_sets:
                If this is true, also sets for couplings will be added.

        Returns:
            A geometry set container that contains all geometry sets of this mesh.
        """

        # Make a copy of the sets in this mesh.
        mesh_sets = self.geometry_sets.copy()

        # Add sets from boundary conditions.
        for (bc_key, geom_key), bc_list in self.boundary_conditions.items():
            for bc in bc_list:
                # Check if sets from couplings should be added.
                is_coupling = bc_key in (
                    _bme.bc.point_coupling,
                    bc_key == _bme.bc.point_coupling_penalty,
                )
                if (is_coupling and coupling_sets) or (not is_coupling):
                    # Only add set if it is not already in the container.
                    # For example if multiple Neumann boundary conditions
                    # are applied on the same node set.
                    if bc.geometry_set not in mesh_sets[geom_key]:
                        mesh_sets[geom_key].append(bc.geometry_set)

        return mesh_sets

    def set_node_links(self):
        """Create a link of all elements to the nodes connected to them."""
        for element in self.elements:
            for node in element.nodes:
                node.element_link.append(element)

    def translate(self, vector: _NDArray | list[float]) -> None:
        """Translate all beam nodes of this mesh.

        Args:
            vector: A 3D vector that will be added to all nodes.
        """
        for node in self.nodes:
            node.coordinates += vector

    def rotate(
        self,
        rotation: _Rotation | _NDArray[_quaternion.quaternion],
        origin=None,
        only_rotate_triads: bool = False,
    ) -> None:
        """Rotate all beam nodes of the mesh with rotation.

        Args:
            rotation: The rotation(s) that will be applied to the nodes. If
            this is an array, it has to hold a quaternion for each node.
            origin (3D vector): If this is given, the mesh is rotated about
                this point. Defaults to (0, 0, 0).
            only_rotate_triads: If this is true, the nodal positions are not
                changed.
        """

        # Get array with all quaternions for the nodes.
        rot1 = _get_nodal_quaternions(self.nodes)

        # Apply the rotation to the rotation of all nodes.
        rot_new = _add_rotations(rotation, rot1)

        if not only_rotate_triads:
            # Get array with all positions for the nodes.
            pos = _get_nodal_coordinates(self.nodes)
            pos_new = _rotate_coordinates(pos, rotation, origin=origin)

        for i, node in enumerate(self.nodes):
            if isinstance(node, _NodeCosserat):
                node.rotation.q = rot_new[i, :]
            if not only_rotate_triads:
                node.coordinates = pos_new[i, :]

    def reflect(self, normal_vector, origin=None, flip_beams: bool = False) -> None:
        """Reflect all nodes of the mesh with respect to a plane defined by its
        normal_vector. Per default the plane goes through the origin, if not a
        point on the plane can be given with the parameter origin.

        For the reflection we assume that e1' and e2' are mirrored with respect
        to the original frame and e3' is in the opposite direction than the
        mirrored e3.

        With the defined mirroring strategy, the quaternion to be applied on
        the existing rotations can be calculated the following way:
            q[0] = e3 * n
            q[1,2,3] = e3 x n
        This constructs a rotation with the rotation axis on the plane, and
        normal to the vector e3. The rotation angle is twice the angle of e3
        to n.

        Args:
            normal_vector (3D vector): The normal vector of the reflection plane.
            origin (3D vector): The reflection plane goes through this point.
                Defaults to (0, 0, 0).
            flip_beams: When True, the beams are flipped, so that the direction
                along the beam is reversed.
        """

        # Normalize the normal vector.
        normal_vector = _np.asarray(normal_vector) / _np.linalg.norm(normal_vector)

        # Get array with all quaternions and positions for the nodes.
        pos = _get_nodal_coordinates(self.nodes)
        rot1 = _get_nodal_quaternions(self.nodes)

        # Check if origin has to be added.
        if origin is not None:
            pos -= origin

        # Get the reflection matrix A.
        A = _np.eye(3) - 2.0 * _np.outer(normal_vector, normal_vector)

        # Calculate the new positions.
        pos_new = _np.dot(pos, A)

        # Move back from the origin.
        if origin is not None:
            pos_new += origin

        # First get all e3 vectors of the nodes.
        e3 = _np.zeros_like(pos)
        e3[:, 0] = 2 * (rot1[:, 0] * rot1[:, 2] + rot1[:, 1] * rot1[:, 3])
        e3[:, 1] = 2 * (-1 * rot1[:, 0] * rot1[:, 1] + rot1[:, 2] * rot1[:, 3])
        e3[:, 2] = rot1[:, 0] ** 2 - rot1[:, 1] ** 2 - rot1[:, 2] ** 2 + rot1[:, 3] ** 2

        # Get the dot and cross product of e3 and the normal vector.
        rot2 = _np.zeros_like(rot1)
        rot2[:, 0] = _np.dot(e3, normal_vector)
        rot2[:, 1:] = _np.cross(e3, normal_vector)

        # Add to the existing rotations.
        rot_new = _add_rotations(rot2, rot1)

        if flip_beams:
            # To achieve the flip, the triads are rotated with the angle pi
            # around the e2 axis.
            rot_flip = _Rotation([0, 1, 0], _np.pi)
            rot_new = _add_rotations(rot_new, rot_flip)

        # For solid elements we need to adapt the connectivity to avoid negative Jacobians.
        # For beam elements this is optional.
        for element in self.elements:
            if isinstance(element, _Beam):
                if flip_beams:
                    element.flip()
            else:
                element.flip()

        # Set the new positions and rotations.
        for i, node in enumerate(self.nodes):
            node.coordinates = pos_new[i, :]
            if isinstance(node, _NodeCosserat):
                node.rotation.q = rot_new[i, :]

    def wrap_around_cylinder(
        self, radius: float | None = None, advanced_warning: bool = True
    ) -> None:
        """Wrap the geometry around a cylinder. The y-z plane gets morphed into
        the z-axis of symmetry. If all nodes are on the same y-z plane, the
        radius of the created cylinder is the x coordinate of that plane. If
        the nodes are not on the same y-z plane, the radius has to be given
        explicitly.

        Args:
            radius: If this value is given AND not all nodes are on the same y-z
                plane, then use this radius for the calculation of phi for all
                nodes. This might still lead to distorted elements!
        advanced_warning: If each element should be checked if it is either parallel
            to the y-z or x-z plane. This is computationally expensive, but in most
            cases (up to 100,000 elements) this check can be left activated.
        """

        pos = _get_nodal_coordinates(self.nodes)
        quaternions = _np.zeros([len(self.nodes), 4])

        # The x coordinate is the radius, the y coordinate the arc length.
        points_x = pos[:, 0].copy()

        # Check if all points are on the same y-z plane.
        if _np.abs(_np.min(points_x) - _np.max(points_x)) > _bme.eps_pos:
            # The points are not all on the y-z plane, get the reference
            # radius.
            if radius is not None:
                if advanced_warning:
                    # Here we check, if each element lays on a plane parallel
                    # to the y-z plane, or parallel to the x-z plane.
                    #
                    # To be exactly sure, we could check the rotations here,
                    # i.e. if they are also in plane.
                    element_warning = []
                    for i_element, element in enumerate(self.elements):
                        element_coordinates = _np.zeros([len(element.nodes), 3])
                        for i_node, node in enumerate(element.nodes):
                            element_coordinates[i_node, :] = node.coordinates
                        is_yz = (
                            _np.max(
                                _np.abs(
                                    element_coordinates[:, 0]
                                    - element_coordinates[0, 0]
                                )
                            )
                            < _bme.eps_pos
                        )
                        is_xz = (
                            _np.max(
                                _np.abs(
                                    element_coordinates[:, 1]
                                    - element_coordinates[0, 1]
                                )
                            )
                            < _bme.eps_pos
                        )
                        if not (is_yz or is_xz):
                            element_warning.append(i_element)
                    if len(element_warning) != 0:
                        _warnings.warn(
                            "There are elements which are not "
                            "parallel to the y-z or x-y plane. This will lead "
                            "to distorted elements!"
                        )
                else:
                    _warnings.warn(
                        "The nodes are not on the same y-z plane. "
                        "This may lead to distorted elements!"
                    )
            else:
                raise ValueError(
                    "The nodes that should be wrapped around a "
                    "cylinder are not on the same y-z plane. This will give "
                    "unexpected results. Give a reference radius!"
                )
            radius_phi = radius
            radius_points = points_x
        elif radius is None or _np.abs(points_x[0] - radius) < _bme.eps_pos:
            radius_points = radius_phi = points_x[0]
        else:
            raise ValueError(
                (
                    "The points are all on the same y-z plane with "
                    "the x-coordinate {} but the given radius {} is different. "
                    "This does not make sense."
                ).format(points_x[0], radius)
            )

        # Get the angle for all nodes.
        phi = pos[:, 1] / radius_phi

        # The rotation is about the z-axis.
        quaternions[:, 0] = _np.cos(0.5 * phi)
        quaternions[:, 3] = _np.sin(0.5 * phi)

        # Set the new positions in the global array.
        pos[:, 0] = radius_points * _np.cos(phi)
        pos[:, 1] = radius_points * _np.sin(phi)

        # Rotate the mesh
        self.rotate(quaternions, only_rotate_triads=True)

        # Set the new position for the nodes.
        for i, node in enumerate(self.nodes):
            node.coordinates = pos[i, :]

    def couple_nodes(
        self,
        *,
        nodes=None,
        reuse_matching_nodes=False,
        coupling_type=_bme.bc.point_coupling,
        coupling_dof_type=_bme.coupling_dof.fix,
    ) -> None:
        """Search through nodes and connect all nodes with the same
        coordinates.

        Args:
            nodes:
                List of nodes to couple. If None is given, all nodes of the mesh
                are coupled (except middle nodes).
            reuse_matching_nodes:
                If two nodes have the same position and rotation, the nodes are
                reduced to one node in the mesh. Be aware, that this might lead to
                issues if not all DOFs of the nodes should be coupled.
            coupling_type:
                Type of point coupling.
            coupling_dof_type:
                `str`: The string that will be used in the input file.
                `bme.coupling_dof.fix`: Fix all positional and rotational DOFs of the
                    nodes together.
                `bme.coupling_dof.joint`: Fix all positional DOFs of the nodes
                    together.
        """

        # Check that a coupling BC is given.
        if coupling_type not in (
            _bme.bc.point_coupling,
            _bme.bc.point_coupling_penalty,
        ):
            raise ValueError(
                "Only coupling conditions can be applied in 'couple_nodes'!"
            )

        # Get the nodes that should be checked for coupling. Middle nodes are
        # not checked, as coupling can only be applied to the boundary nodes.
        if nodes is None:
            node_list = self.nodes
        else:
            node_list = nodes
        node_list = _filter_nodes(node_list, middle_nodes=False)
        partner_nodes = _find_close_nodes(node_list)
        if len(partner_nodes) == 0:
            # If no partner nodes were found, end this function.
            return

        if reuse_matching_nodes:
            # Check if there are nodes with the same rotation. If there are the
            # nodes are reused, and no coupling is inserted.

            # Go through partner nodes.
            node_replacement_map: dict[_Node, _Node] = {}
            for partner_node_list in partner_nodes:
                # Get array with rotation vectors.
                rotation_vectors = _np.zeros([len(partner_node_list), 3])
                for i, node in enumerate(partner_node_list):
                    if isinstance(node, _NodeCosserat):
                        rotation_vectors[i, :] = node.rotation.get_rotation_vector()
                    else:
                        # For the case of nodes that belong to solid elements,
                        # we define the following default value:
                        rotation_vectors[i, :] = [4 * _np.pi, 0, 0]

                # Use find close points function to find nodes with the
                # same rotation.
                partners, n_partners = _find_close_points(
                    rotation_vectors, tol=_bme.eps_quaternion
                )

                # Check if nodes with the same rotations were found.
                if n_partners == 0:
                    self.add(
                        _coupling_factory(
                            partner_node_list, coupling_type, coupling_dof_type
                        )
                    )
                else:
                    # There are nodes that need to be combined.
                    combining_nodes: list[list[_Node]] = []
                    coupling_nodes: list[_Node] = []
                    found_partner_id: list[int | None] = [
                        None for _i in range(n_partners)
                    ]

                    # Add the nodes that need to be combined and add the nodes
                    # that will be coupled.
                    for i, partner in enumerate(partners):
                        if partner == -1:
                            # This node does not have a partner with the same
                            # rotation.
                            coupling_nodes.append(partner_node_list[i])

                        elif found_partner_id[partner] is not None:
                            # This node has already a processed partner, add
                            # this one to the combining nodes.
                            combining_nodes[found_partner_id[partner]].append(
                                partner_node_list[i]
                            )

                        else:
                            # This is the first node of a partner set that was
                            # found. This one will remain, the other ones will
                            # be replaced with this one.
                            new_index = len(combining_nodes)
                            found_partner_id[partner] = new_index
                            combining_nodes.append([partner_node_list[i]])
                            coupling_nodes.append(partner_node_list[i])

                    # Add the coupling nodes.
                    if len(coupling_nodes) > 1:
                        self.add(
                            _coupling_factory(
                                coupling_nodes, coupling_type, coupling_dof_type
                            )
                        )

                    # Add to the replacement map.
                    for combine_list in combining_nodes:
                        target_node = combine_list[0]
                        for node in combine_list[1:]:
                            node_replacement_map[node] = target_node

            # Replace the nodes in the elements and geometry sets.
            self.replace_nodes(node_replacement_map)

        else:
            # Connect close nodes with a coupling.
            for node_list in partner_nodes:
                self.add(_coupling_factory(node_list, coupling_type, coupling_dof_type))

    def unlink_nodes(self):
        """Delete the linked arrays and global indices in all nodes."""
        for node in self.nodes:
            node.unlink()

    def get_nodes_by_function(self, *args, **kwargs):
        """Return all nodes for which the function evaluates to true."""
        return _get_nodes_by_function(self.nodes, *args, **kwargs)

    def get_mesh_representation(
        self, material_to_i_global: dict[_Material, int] | None = None
    ) -> tuple[
        _MeshRepresentation,
        dict[int, _Any],
        dict[_GeometrySetBase, int],
        dict[_NURBSPatch, int],
    ]:
        """Create a mesh representation for this mesh.

        This function does not alter the mesh object. It assigns internal IDs to the
        mesh object and returns mappings between the objects and the IDs.

        Args:
            material_to_i_global: A dictionary that maps materials to their global
                index in the mesh representation.

        Returns:
            mesh_representation: `MeshRepresentation` object for this mesh.
            element_type_id_to_data: A dictionary that maps the element type id to the
                data of the element type.
            geometry_sets_to_i_global: A dictionary that maps geometry sets to their
                global index in the mesh representation.
            nurbs_patch_to_i_global: A dictionary that maps each NURBS patch to the
                global ID of that patch.
        """

        if material_to_i_global is None:
            material_to_i_global = {}

        # Get the global id mappings for geometry sets.
        mesh_sets = self.get_unique_geometry_sets()
        geometry_sets_to_i_global: dict[_GeometrySetBase, int] = {}
        for geometry_type, geometry_list in mesh_sets.items():
            for geometry_set in geometry_list:
                geometry_sets_to_i_global[geometry_set] = len(geometry_sets_to_i_global)

        # Extract nodes for the mesh representation and assign global IDs. We also extract
        # other required information like the node type, the nodal rotation vector and
        # control point weights information here. The weights are optional, so the array
        # might be `None`.
        n_nodes = len(self.nodes)
        if n_nodes != len(set(self.nodes)):
            raise ValueError("Nodes are not unique!")
        points = _np.zeros((n_nodes, 3))
        point_types = _np.full(n_nodes, -1)
        control_point_weights = None
        for i_node, node in enumerate(self.nodes):
            node.i_global = i_node
            node_type = type(node).node_type
            point_types[i_node] = node_type.value
            points[i_node] = node.coordinates
            if node_type == _bme.node_type.control_point:
                if control_point_weights is None:
                    control_point_weights = _np.full(n_nodes, -1.0)
                control_point_weights[i_node] = node.weight
        # We don't get the rotation vectors in the loop, that would be very slow, instead
        # we get the global quaternion array and convert that directly using the numpy
        # quaternion library.
        nodal_quaternions = _quaternion.from_float_array(
            _get_nodal_quaternions(self.nodes)
        )
        nodal_rotation_vectors = _quaternion.as_rotation_vector(nodal_quaternions)

        # Check that element are unique.
        if len(self.elements) != len(set(self.elements)):
            raise ValueError("Elements are not unique!")

        # For the elements, we first have to loop over all elements, so we get the total
        # number of elements, as NURBS patches can contain multiple elements.
        # This is needed to initialize the numpy arrays with the correct size.
        i_element = 0
        nurbs_count = 0
        nurbs_patch_to_i_global = {}
        for element in self.elements:
            # Perform consistency checks for the element.
            element.check()
            element.i_global = i_element
            if isinstance(element, _NURBSPatch):
                nurbs_patch_to_i_global[element] = nurbs_count
                nurbs_count += 1
                i_element += element.get_number_of_elements()
            else:
                i_element += 1
        n_elements = i_element

        # Now that we know the expected size, we can allocate the data arrays and
        # actually gather the element data.
        element_type_to_id: dict[type, int] = {}
        element_type_id_to_data: dict[int, _Any] = {}
        cell_connectivity = []
        cell_types = _np.full(n_elements, -1)
        cell_element_type_ids = _np.full(n_elements, -1)
        cell_material_ids = _np.full(n_elements, -1)
        cell_beamme_element_ids = _np.full(n_elements, -1)
        for i_element_beamme, element in enumerate(self.elements):
            # Get the element type id for this element.
            if type(element) not in element_type_to_id:
                element_type_id = len(element_type_to_id)
                element_type_to_id[type(element)] = element_type_id
                element_type_id_to_data[element_type_id] = _copy.deepcopy(
                    type(element).data
                )
            else:
                element_type_id = element_type_to_id[type(element)]

            # For elements which don't require a material we set the material id to -1.
            # This is currently only the case for rigid sphere elements in 4C.
            material_id = material_to_i_global.get(element.material, -1)

            if isinstance(element, _NURBSPatch):
                n_patch_elements = element.get_number_of_elements()
                # To satisfy mypy, we do a cast here, since we know that we have set
                # i_global previously for all elements.
                element_i_global = _cast(int, element.i_global)
                data_assignment_slice = slice(
                    element_i_global, element_i_global + n_patch_elements
                )

                for knot_span in element.get_knot_span_iterator():
                    element_cps_ids = element.get_ids_ctrlpts(*knot_span)
                    connectivity = [
                        element.nodes[index].i_global for index in element_cps_ids
                    ]
                    cell_connectivity.extend([len(connectivity), *connectivity])

            else:
                data_assignment_slice = element.i_global

                reorder_indices = _MESH_REPRESENTATION_MAPPINGS[
                    "element_type_and_n_nodes_to_connectivity_mapping_beamme_to_vtk"
                ].get((type(element).element_type, len(element.nodes)), None)
                if reorder_indices is not None:
                    connectivity = [
                        element.nodes[index].i_global for index in reorder_indices
                    ]
                else:
                    connectivity = [node.i_global for node in element.nodes]

                cell_connectivity.extend([len(connectivity), *connectivity])

            cell_material_ids[data_assignment_slice] = material_id
            cell_element_type_ids[data_assignment_slice] = element_type_id
            cell_types[data_assignment_slice] = type(element).vtk_cell_type
            cell_beamme_element_ids[data_assignment_slice] = i_element_beamme

        # Extract geometry sets.
        geometry_sets = []
        for geometry_type, geometry_list in mesh_sets.items():
            for geometry_set in geometry_list:
                node_set_flag = _np.zeros(n_nodes, dtype=int)
                node_set_flag[
                    [node.i_global for node in geometry_set.get_all_nodes()]
                ] = 1
                if isinstance(geometry_set, _GeometrySet) and (
                    geometry_type == _bme.geo.line
                    or geometry_type == _bme.geo.surface
                    or geometry_type == _bme.geo.volume
                ):
                    element_set_flag = _np.zeros(n_elements, dtype=int)
                    element_set_indices: list[int] = []
                    for element in geometry_set.get_geometry_objects():
                        if isinstance(element, _NURBSPatch):
                            # For NURBS, we have to set the flag for all elements that are part of the patch.
                            element_i_global = _cast(int, element.i_global)
                            element_set_indices.extend(
                                range(
                                    element_i_global,
                                    element_i_global + element.get_number_of_elements(),
                                )
                            )
                        else:
                            element_set_indices.append(element.i_global)
                    element_set_flag[element_set_indices] = 1

                else:
                    element_set_flag = None
                geometry_set_wrapper = _GeometrySetInfo(
                    geometry_type=geometry_type,
                    i_global=geometry_sets_to_i_global[geometry_set],
                    point_flag_vector=node_set_flag,
                    cell_flag_vector=element_set_flag,
                    name=geometry_set.name,
                )
                geometry_sets.append(geometry_set_wrapper)

        # Reset the previously set indices.
        for node in self.nodes:
            node.i_global = None
        for element in self.elements:
            element.i_global = None

        # Create the mesh representation.
        mesh_representation = _MeshRepresentation(
            cell_connectivity=cell_connectivity,
            cell_types=cell_types,
            points=points,
            geometry_sets=geometry_sets,
            cell_data={
                "element_type_id": cell_element_type_ids,
                "material_id": cell_material_ids,
                "beamme_id": cell_beamme_element_ids,
            },
            point_data={
                "point_type": point_types,
                "control_point_weight": control_point_weights,
                "rotation_vector": nodal_rotation_vectors,
            },
        )

        return (
            mesh_representation,
            element_type_id_to_data,
            geometry_sets_to_i_global,
            nurbs_patch_to_i_global,
        )

    def get_vtu_representation(self) -> _pv.UnstructuredGrid:
        """Return a vtu representation of this mesh.

        Returns:
            A pyvista UnstructuredGrid object that represents this mesh.
        """

        # Get mesh representation.
        mesh_representation, _, _, _ = self.get_mesh_representation()

        # Get data arrays for visualization, i.e., element type and the cross-section radius for beams.
        element_types = _np.empty(mesh_representation.n_cells, dtype=int)
        cross_section_radii = _np.zeros(mesh_representation.n_cells)
        for i_cell, beamm_id in enumerate(
            mesh_representation.data_iterator("cell_data", "beamme_id")
        ):
            element = self.elements[beamm_id]
            element_type = type(element).element_type
            element_types[i_cell] = element_type.value
            if element_type == _bme.element_type.beam:
                cross_section_radii[i_cell] = element.material.radius
        node_value = _np.zeros(mesh_representation.n_points)
        for i_node, node in enumerate(self.nodes):
            if isinstance(node, _NodeCosserat):
                if node.is_middle_node:
                    node_value[i_node] = 0.5
                else:
                    node_value[i_node] = 1.0

        # Create a pyvista grid.
        grid = _pv.UnstructuredGrid(
            mesh_representation.cell_connectivity,
            mesh_representation.cell_types,
            mesh_representation.points,
        )

        # Add all data arrays required from the mesh representation
        data_names_for_visualization = {
            "cell_data": ["element_type_id", "beamme_id"],
            "point_data": ["point_type", "rotation_vector"],
        }
        for field_name in ["cell_data", "point_data"]:
            for name, data in getattr(mesh_representation, field_name).items():
                geometry_set_info = _string_to_geometry_set_info(name)
                if (
                    name in data_names_for_visualization[field_name]
                    or geometry_set_info is not None
                ):
                    getattr(grid, field_name)[name] = data

        # Add the data arrays created here.
        grid.cell_data["element_type"] = element_types
        grid.cell_data["cross_section_radius"] = cross_section_radii
        grid.point_data["node_value"] = node_value

        return grid

    def write_vtu(self, file_name: _Path | str, binary=True):
        """Write the contents of this mesh to VTK files.

        Args
        ----
        file_name: The path or filename of the vtu file.
        binary: If the data should be written encoded in binary or in human readable text
        """
        path = _Path(file_name)
        if path.suffix == "":
            path = path.with_suffix(".vtu")
        elif path.suffix != ".vtu":
            raise ValueError(f"Expected file extension '.vtu', got '{path.suffix}'")
        grid = self.get_vtu_representation()
        grid.save(path, binary=binary)

    def display_pyvista(
        self,
        *,
        beam_nodes=True,
        beam_tube=True,
        beam_cross_section_directors=True,
        resolution=20,
        parallel_projection=False,
    ):
        """Display the mesh in pyvista.

        If this is called in a GitHub testing run, nothing will be shown, instead
        the _pv.plotter object will be returned.

        Args
        ----
        beam_nodes: bool
            If the beam nodes should be displayed. The start and end nodes of each
            beam will be shown in green, possible middle nodes inside the element
            are shown in cyan.
        beam_tube: bool
            If the beam should be rendered as a tube
        beam_cross_section_directors: bool
            If the cross section directors should be displayed (at each node)
        resolution: int
            Indicates how many triangulations will be performed to visualize arrows,
            tubes and spheres.
        parallel_projection: bool
            Flag to change camera view to parallel projection.
        """

        grid = self.get_vtu_representation()

        plotter = _pv.Plotter()
        plotter.renderer.add_axes()

        if parallel_projection:
            plotter.enable_parallel_projection()

        beam_mask = grid.cell_data["element_type"] == _bme.element_type.beam.value
        if _np.any(beam_mask):
            beam_grid = grid.extract_cells(beam_mask).cell_data_to_point_data()

            # Plot the nodes
            beam_finite_element_nodes = beam_grid.cast_to_poly_points()
            node_radius_scaling_factor = 1.5
            if beam_nodes:
                sphere = _pv.Sphere(
                    radius=1.0,
                    theta_resolution=resolution,
                    phi_resolution=resolution,
                )
                nodes_glyph = beam_finite_element_nodes.glyph(
                    geom=sphere,
                    scale="cross_section_radius",
                    factor=node_radius_scaling_factor,
                    orient=False,
                )
                plotter.add_mesh(
                    nodes_glyph.threshold(scalars="node_value", value=(0.9, 1.1)),
                    color="green",
                )
                middle_nodes = nodes_glyph.threshold(
                    scalars="node_value", value=(0.4, 0.6)
                )
                if len(middle_nodes.points) > 0:
                    plotter.add_mesh(middle_nodes, color="cyan")

            # Plot the beams
            beam_color = [0.5, 0.5, 0.5]
            if beam_tube:
                beam_tube_grid = beam_grid.extract_surface()
                beam_tube_grid = beam_tube_grid.tube(
                    scalars="cross_section_radius",
                    absolute=True,
                    n_sides=resolution,
                )
                plotter.add_mesh(beam_tube_grid, color=beam_color)
            else:
                plotter.add_mesh(beam_grid, color=beam_color, line_width=4)

            # Plot the directors of the beam cross-section
            if beam_cross_section_directors:
                # First, we need to set the cross section basis vectors as point data vectors.
                quaternions = _quaternion.from_rotation_vector(
                    beam_finite_element_nodes["rotation_vector"]
                )
                rotation_matrices = _quaternion.as_rotation_matrix(quaternions)
                for i in range(3):
                    beam_finite_element_nodes[f"base_vector_{i + 1}"] = (
                        rotation_matrices[:, :, i]
                    )

                director_radius_scaling_factor = 3.5
                arrow = _pv.Arrow(
                    tip_resolution=resolution, shaft_resolution=resolution
                )
                directors = [
                    beam_finite_element_nodes.glyph(
                        geom=arrow,
                        orient=f"base_vector_{i + 1}",
                        scale="cross_section_radius",
                        factor=director_radius_scaling_factor,
                    )
                    for i in range(3)
                ]
                colors = ["white", "blue", "red"]
                for i, arrow in enumerate(directors):
                    plotter.add_mesh(arrow, color=colors[i])

        solid_mask = grid.cell_data["element_type"] == _bme.element_type.beam.solid
        if _np.any(solid_mask):
            solid_grid = grid.extract_cells(solid_mask)
            plotter.add_mesh(solid_grid, color="white", show_edges=True, opacity=0.5)

        if not _is_testing():
            plotter.show()
        else:
            return plotter

    def copy(self) -> "Mesh":
        """Return a deep copy of this mesh.

        The internal mesh data (nodes, elements, boundary conditions, and
        geometry sets) are deep-copied. Materials and functions are not
        deep-copied.

        **Important:** Some mesh creation functions return geometry set
        containers (e.g., node or element sets) that hold a reference to the
        nodes or elements of the mesh they were created with. When using
        ``mesh.copy()``, these externally returned sets remain linked to the
        original mesh and are therefore not transferred to the copied mesh.

        To copy both the mesh and the corresponding geometry sets correctly,
        deep-copy them together.

        Returns:
            A deep copy of the mesh.
        """
        return _copy.deepcopy(self)

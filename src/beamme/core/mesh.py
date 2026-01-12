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
"""This module defines the Mesh class, which holds the content (nodes,
elements, sets, ...) for a meshed geometry."""

import copy as _copy
import os as _os
import warnings as _warnings
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional

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
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.geometry_set import GeometrySetContainer as _GeometrySetContainer
from beamme.core.material import Material as _Material
from beamme.core.node import Node as _Node
from beamme.core.node import NodeCosserat as _NodeCosserat
from beamme.core.rotation import Rotation as _Rotation
from beamme.core.rotation import add_rotations as _add_rotations
from beamme.core.rotation import rotate_coordinates as _rotate_coordinates
from beamme.core.vtk_writer import VTKWriter as _VTKWriter
from beamme.geometric_search.find_close_points import (
    find_close_points as _find_close_points,
)
from beamme.geometric_search.find_close_points import (
    point_partners_to_partner_indices as _point_partners_to_partner_indices,
)
from beamme.utils.environment import is_testing as _is_testing
from beamme.utils.nodes import filter_nodes as _filter_nodes
from beamme.utils.nodes import find_close_nodes as _find_close_nodes
from beamme.utils.nodes import get_min_max_nodes as _get_min_max_nodes
from beamme.utils.nodes import get_nodal_coordinates as _get_nodal_coordinates
from beamme.utils.nodes import get_nodal_quaternions as _get_nodal_quaternions
from beamme.utils.nodes import get_nodes_by_function as _get_nodes_by_function
from beamme.utils.visualization import show_plotter as _show_plotter


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

    def replace_node(self, old_node, new_node):
        """Replace the first node with the second node."""

        # Check that the new node is in mesh.
        if new_node not in self.nodes:
            raise ValueError("The new node is not in the mesh!")

        for i, node in enumerate(self.nodes):
            if node == old_node:
                del self.nodes[i]
                break
        else:
            raise ValueError("The node that should be replaced is not in the mesh")

    def get_unique_geometry_sets(
        self,
        *,
        coupling_sets: bool = True,
        link_to_nodes: str = "no_link",
        geometry_set_start_indices: _Optional[_Dict] = None,
    ) -> _GeometrySetContainer:
        """Return a geometry set container that contains geometry sets
        explicitly added to the mesh, as well as sets for boundary conditions.

        The i_global values are set in the returned geometry sets.

        Args:
            coupling_sets:
                If this is true, also sets for couplings will be added.
            link_to_nodes:
                "no_link":
                    No link between the geometry set and the nodes is set
                "explicitly_contained_nodes":
                    A link will be set for all nodes that are explicitly part of the geometry set
                "all_nodes":
                    A link will be set for all nodes that are part of the geometry set, i.e., also
                    nodes connected to elements of an element set. This is mainly used for vtk
                    output so we can color the nodes which are part of element sets.
            geometry_set_start_indices:
                Dictionary where the keys are geometry type and the value is the starting
                index for those geometry sets. This can be used to define an offset in the
                i_global numbering of the geometry sets. The offsets default to 0.
        """

        is_link_nodes = not link_to_nodes == "no_link"
        if is_link_nodes:
            # First clear all links in existing nodes.
            for node in self.nodes:
                node.node_sets_link = []

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

        for key in mesh_sets.keys():
            i_global_offset = 0
            if geometry_set_start_indices is not None:
                if key in geometry_set_start_indices:
                    i_global_offset = geometry_set_start_indices[key]
                else:
                    raise KeyError("Could not find {key} in geometry_set_start_indices")
            for i, geometry_set in enumerate(mesh_sets[key]):
                # Add global indices to the geometry set.
                geometry_set.i_global = i + i_global_offset
                if is_link_nodes:
                    geometry_set.link_to_nodes(link_to_nodes=link_to_nodes)

        return mesh_sets

    def set_node_links(self):
        """Create a link of all elements to the nodes connected to them.

        Also add a link to this mesh.
        """
        for element in self.elements:
            for node in element.nodes:
                node.element_link.append(element)
        for node in self.nodes:
            node.mesh = self

    def translate(self, vector):
        """Translate all beam nodes of this mesh.

        Args
        ----
        vector: _np.array, list
            3D vector that will be added to all nodes.
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
    ):
        """Search through nodes and connect all nodes with the same
        coordinates.

        Args:
        ----
        nodes: [Node]
            List of nodes to couple. If None is given, all nodes of the mesh
            are coupled (except middle nodes).
        reuse_matching_nodes: bool
            If two nodes have the same position and rotation, the nodes are
            reduced to one node in the mesh. Be aware, that this might lead to
            issues if not all DOFs of the nodes should be coupled.
        coupling_type: bme.bc
            Type of point coupling.
        coupling_dof_type: str, bme.coupling_dof
            str: The string that will be used in the input file.
            bme.coupling_dof.fix: Fix all positional and rotational DOFs of the
                nodes together.
            bme.coupling_dof.joint: Fix all positional DOFs of the nodes
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

            # Set the links to all nodes in the mesh. In this case we have to use
            # "all_nodes" since we also have to replace nodes that are in existing
            # GeometrySetNodes.
            self.unlink_nodes()
            self.get_unique_geometry_sets(link_to_nodes="explicitly_contained_nodes")
            self.set_node_links()

            # Go through partner nodes.
            for node_list in partner_nodes:
                # Get array with rotation vectors.
                rotation_vectors = _np.zeros([len(node_list), 3])
                for i, node in enumerate(node_list):
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
                        _coupling_factory(node_list, coupling_type, coupling_dof_type)
                    )
                else:
                    # There are nodes that need to be combined.
                    combining_nodes = []
                    coupling_nodes = []
                    found_partner_id = [None for _i in range(n_partners)]

                    # Add the nodes that need to be combined and add the nodes
                    # that will be coupled.
                    for i, partner in enumerate(partners):
                        if partner == -1:
                            # This node does not have a partner with the same
                            # rotation.
                            coupling_nodes.append(node_list[i])

                        elif found_partner_id[partner] is not None:
                            # This node has already a processed partner, add
                            # this one to the combining nodes.
                            combining_nodes[found_partner_id[partner]].append(
                                node_list[i]
                            )

                        else:
                            # This is the first node of a partner set that was
                            # found. This one will remain, the other ones will
                            # be replaced with this one.
                            new_index = len(combining_nodes)
                            found_partner_id[partner] = new_index
                            combining_nodes.append([node_list[i]])
                            coupling_nodes.append(node_list[i])

                    # Add the coupling nodes.
                    if len(coupling_nodes) > 1:
                        self.add(
                            _coupling_factory(
                                coupling_nodes, coupling_type, coupling_dof_type
                            )
                        )

                    # Replace the identical nodes.
                    for combine_list in combining_nodes:
                        master_node = combine_list[0]
                        for node in combine_list[1:]:
                            node.replace_with(master_node)

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

    def get_min_max_nodes(self, *args, **kwargs):
        """Return a geometry set with the max and min nodes in all
        directions."""
        return _get_min_max_nodes(self.nodes, *args, **kwargs)

    def check_overlapping_elements(self, raise_error=True):
        """Check if there are overlapping elements in the mesh.

        This is done by checking if all middle nodes of beam elements
        have unique coordinates in the mesh.
        """

        # Number of middle nodes.
        middle_nodes = [node for node in self.nodes if node.is_middle_node]

        # Only check if there are middle nodes.
        if len(middle_nodes) == 0:
            return

        # Get array with middle nodes.
        coordinates = _np.zeros([len(middle_nodes), 3])
        for i, node in enumerate(middle_nodes):
            coordinates[i, :] = node.coordinates

        # Check if there are double entries in the coordinates.
        has_partner, partner = _find_close_points(coordinates)
        partner_indices = _point_partners_to_partner_indices(has_partner, partner)
        if partner > 0:
            if raise_error:
                raise ValueError(
                    "There are multiple middle nodes with the "
                    "same coordinates. Per default this raises an error! "
                    "This check can be turned of with "
                    "bme.check_overlapping_elements=False"
                )
            else:
                _warnings.warn(
                    "There are multiple middle nodes with the same coordinates!"
                )

            # Add the partner index to the middle nodes.
            for i_partner, partners in enumerate(partner_indices):
                for i_node in partners:
                    middle_nodes[i_node].element_partner_index = i_partner

    def get_vtk_representation(
        self, *, overlapping_elements=True, coupling_sets=False, **kwargs
    ):
        """Return a vtk representation of the beams and solid in this mesh.

        Args
        ----
        overlapping_elements: bool
            I elements should be checked for overlapping. If they overlap, the
            output will mark them.
        coupling_sets: bool
            If coupling sets should also be displayed.
        """

        # Object to store VKT data (and write it to file)
        vtk_writer_beam = _VTKWriter()
        vtk_writer_solid = _VTKWriter()

        # Get the set numbers of the mesh
        mesh_sets = self.get_unique_geometry_sets(
            coupling_sets=coupling_sets, link_to_nodes="all_nodes"
        )

        # Set the global value for digits in the VTK output.
        # Get highest number of node_sets.
        max_sets = max(len(geometry_list) for geometry_list in mesh_sets.values())

        # Set the bme value.
        digits = len(str(max_sets))
        _bme.vtk_node_set_format = "{:0" + str(digits) + "}"

        if overlapping_elements:
            # Check for overlapping elements.
            self.check_overlapping_elements(raise_error=False)

        # Get representation of elements.
        for element in self.elements:
            element.get_vtk(vtk_writer_beam, vtk_writer_solid, **kwargs)

        # Finish and return the writers
        vtk_writer_beam.complete_data()
        vtk_writer_solid.complete_data()
        return vtk_writer_beam, vtk_writer_solid

    def write_vtk(
        self, output_name="beamme", output_directory="", binary=True, **kwargs
    ):
        """Write the contents of this mesh to VTK files.

        Args
        ----
        output_name: str
            Base name of the output file. There will be a {}_beam.vtu and
            {}_solid.vtu file.
        output_directory: path
            Directory where the output files will be written.
        binary: bool
            If the data should be written encoded in binary or in human readable text

        **kwargs
            For all of them look into:
                Mesh().get_vtk_representation
                Beam().get_vtk
                VolumeElement().get_vtk
        ----
        beam_centerline_visualization_segments: int
            Number of segments to be used for visualization of beam centerline between successive
            nodes. Default is 1, which means a straight line is drawn between the beam nodes. For
            Values greater than 1, a Hermite interpolation of the centerline is assumed for
            visualization purposes.
        """

        vtk_writer_beam, vtk_writer_solid = self.get_vtk_representation(**kwargs)

        # Write to file, only if there is at least one point in the writer.
        if vtk_writer_beam.points.GetNumberOfPoints() > 0:
            filepath = _os.path.join(output_directory, output_name + "_beam.vtu")
            vtk_writer_beam.write_vtk(filepath, binary=binary)
        if vtk_writer_solid.points.GetNumberOfPoints() > 0:
            filepath = _os.path.join(output_directory, output_name + "_solid.vtu")
            vtk_writer_solid.write_vtk(filepath, binary=binary)

    def display_pyvista(
        self,
        *,
        beam_nodes=True,
        beam_tube=True,
        beam_cross_section_directors=True,
        beam_radius_for_display=None,
        resolution=20,
        parallel_projection=False,
        **kwargs,
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
        beam_radius_for_display: float
            If not all beams have an explicitly given radius (in the material
            definition) this value will be used to approximate the beams radius
            for visualization
        resolution: int
            Indicates how many triangulations will be performed to visualize arrows,
            tubes and spheres.
        parallel_projection: bool
            Flag to change camera view to parallel projection.

        **kwargs
            For all of them look into:
                Mesh().get_vtk_representation
                Beam().get_vtk
                VolumeElement().get_vtk
        ----
        beam_centerline_visualization_segments: int
            Number of segments to be used for visualization of beam centerline between successive
            nodes. Default is 1, which means a straight line is drawn between the beam nodes. For
            Values greater than 1, a Hermite interpolation of the centerline is assumed for
            visualization purposes.
        """

        plotter = _pv.Plotter()
        plotter.renderer.add_axes()

        if parallel_projection:
            plotter.enable_parallel_projection()

        vtk_writer_beam, vtk_writer_solid = self.get_vtk_representation(**kwargs)

        if vtk_writer_beam.points.GetNumberOfPoints() > 0:
            beam_grid = _pv.UnstructuredGrid(vtk_writer_beam.grid)

            # Check if all beams have a given cross-section radius, if not set the given input
            # value
            all_beams_have_cross_section_radius = (
                min(beam_grid.cell_data["cross_section_radius"]) > 0
            )
            if not all_beams_have_cross_section_radius:
                if beam_radius_for_display is None:
                    raise ValueError(
                        "Not all beams have a radius, you need to set "
                        "beam_radius_for_display to allow a display of the beams."
                    )
                beam_grid.cell_data["cross_section_radius"] = beam_radius_for_display

            # Grid with beam polyline
            beam_grid = beam_grid.cell_data_to_point_data()

            # Poly data for nodes
            finite_element_nodes = beam_grid.cast_to_poly_points().threshold(
                scalars="node_value", value=(0.4, 1.1)
            )

            # Plot the nodes
            node_radius_scaling_factor = 1.5
            if beam_nodes:
                sphere = _pv.Sphere(
                    radius=1.0,
                    theta_resolution=resolution,
                    phi_resolution=resolution,
                )
                nodes_glyph = finite_element_nodes.glyph(
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
                surface = beam_grid.extract_surface()
                if all_beams_have_cross_section_radius:
                    tube = surface.tube(
                        scalars="cross_section_radius",
                        absolute=True,
                        n_sides=resolution,
                    )
                else:
                    tube = surface.tube(
                        radius=beam_radius_for_display, n_sides=resolution
                    )
                plotter.add_mesh(tube, color=beam_color)
            else:
                plotter.add_mesh(beam_grid, color=beam_color, line_width=4)

            # Plot the directors of the beam cross-section
            director_radius_scaling_factor = 3.5
            if beam_cross_section_directors:
                arrow = _pv.Arrow(
                    tip_resolution=resolution, shaft_resolution=resolution
                )
                directors = [
                    finite_element_nodes.glyph(
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

        if vtk_writer_solid.points.GetNumberOfPoints() > 0:
            solid_grid = _pv.UnstructuredGrid(vtk_writer_solid.grid).clean()
            plotter.add_mesh(solid_grid, color="white", show_edges=True, opacity=0.5)

        if not _is_testing():
            _show_plotter(plotter)
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

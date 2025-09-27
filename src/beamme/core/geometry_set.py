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
"""This module implements a basic class to manage geometry in the input
file."""

from typing import KeysView as _KeysView
from typing import Sequence as _Sequence
from typing import Union as _Union
from typing import cast as _cast

import beamme.core.conf as _conf
from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem
from beamme.core.conf import bme as _bme
from beamme.core.container import ContainerBase as _ContainerBase
from beamme.core.element import Element as _Element
from beamme.core.element_beam import Beam as _Beam
from beamme.core.node import Node as _Node


class GeometrySetBase(_BaseMeshItem):
    """Base class for a geometry set."""

    def __init__(
        self, geometry_type: _conf.Geometry, name: str | None = None, **kwargs
    ):
        """Initialize the geometry set.

        Args:
            geometry_type: Type of geometry. Only geometry sets of a single specified geometry type are supported.
            name: Optional name to identify this geometry set.
        """
        super().__init__(**kwargs)

        self.geometry_type = geometry_type
        self.name = name

    def link_to_nodes(
        self, *, link_to_nodes: str = "explicitly_contained_nodes"
    ) -> None:
        """Set a link to this object in the all contained nodes of this
        geometry set.

        Args:
            link_to_nodes:
                "explicitly_contained_nodes":
                    A link will be set for all nodes that are explicitly part of the geometry set
                "all_nodes":
                    A link will be set for all nodes that are part of the geometry set, i.e., also
                    nodes connected to elements of an element set. This is mainly used for vtk
                    output so we can color the nodes which are part of element sets.
        """
        node_list: list[_Node] | _KeysView[_Node]
        if link_to_nodes == "explicitly_contained_nodes":
            node_list = self.get_node_dict().keys()
        elif link_to_nodes == "all_nodes":
            node_list = self.get_all_nodes()
        else:
            raise ValueError(f'Got unexpected value link nodes="{link_to_nodes}"')
        for node in node_list:
            node.node_sets_link.append(self)

    def check_replaced_nodes(self) -> None:
        """Check if nodes in this set have to be replaced.

        We need to do this for explicitly contained nodes in this set.
        """
        # Don't iterate directly over the keys as the dict changes during this iteration
        for node in list(self.get_node_dict().keys()):
            if node.master_node is not None:
                self.replace_node(node, node.get_master_node())

    def replace_node(self, old_node: _Node, new_node: _Node) -> None:
        """Replace an existing node in this geometry set with a new one.

        Args:
            old_node: Node to be replaced.
            new_node: Node that will be placed instead of old_node.
        """

        explicit_nodes_in_this_set = self.get_node_dict()
        explicit_nodes_in_this_set[new_node] = None
        del explicit_nodes_in_this_set[old_node]

    def get_node_dict(self) -> dict[_Node, None]:
        """Determine the explicitly added nodes for this set, i.e., nodes
        contained in elements are not returned.

        Returns:
            A dictionary containing the explicitly added nodes for this set.
        """
        raise NotImplementedError(
            'The "get_node_dict" method has to be overwritten in the derived class'
        )

    def get_points(self) -> list[_Node]:
        """Determine all points (represented by nodes) for this set.

        This function only works for point sets.

        Returns:
            A list containing the points (represented by nodes) associated with this set.
        """
        raise NotImplementedError(
            'The "get_points" method has to be overwritten in the derived class'
        )

    def get_all_nodes(self) -> list[_Node]:
        """Determine all nodes associated with this set.

        This includes nodes contained within the geometry added to this
        set, e.g., nodes connected to elements in element sets.

        Returns:
            A list containing all associated nodes.
        """
        raise NotImplementedError(
            'The "get_all_nodes" method has to be overwritten in the derived class'
        )

    def __add__(self, other):
        """Create a new geometry set with the combined geometries from this set
        and the other set.

        Args:
            other: Geometry set to be added to this one. This has to be of the same geometry type as this set.
        Returns:
            A combined geometry set.
        """
        combined_set = self.copy()
        combined_set.add(other)
        return combined_set


class GeometrySet(GeometrySetBase):
    """Geometry set which is defined by geometric entries."""

    def __init__(
        self,
        geometry: _Node | _Element | _Sequence[_Node | _Element] | "GeometrySet",
        **kwargs,
    ):
        """Initialize the geometry set.

        Args:
            geometry: Geometry entries to be contained in this set.
        """

        # This is ok, we check every single type in the add method
        if isinstance(geometry, list):
            geometry_type = self._get_geometry_type(geometry[0])
        else:
            geometry_type = self._get_geometry_type(geometry)

        super().__init__(geometry_type, **kwargs)

        self.geometry_objects: dict[_conf.Geometry, dict[_Node | _Element, None]] = {}
        for geo in _bme.geo:
            self.geometry_objects[geo] = {}
        self.add(geometry)

    @staticmethod
    def _get_geometry_type(
        item: _Node | _Element | _Sequence[_Node | _Element] | "GeometrySet",
    ) -> _conf.Geometry:
        """Get the geometry type of a given item.

        Returns:
            Geometry type of the geometry set.
        """

        if isinstance(item, _Node):
            return _bme.geo.point
        elif isinstance(item, _Beam):
            return _bme.geo.line
        elif isinstance(item, GeometrySet):
            return item.geometry_type
        raise TypeError(f"Got unexpected type {type(item)}")

    def add(
        self, item: _Node | _Element | _Sequence[_Node | _Element] | "GeometrySet"
    ) -> None:
        """Add geometry item(s) to this object."""

        if isinstance(item, list):
            for sub_item in item:
                self.add(sub_item)
        elif isinstance(item, GeometrySet):
            if item.geometry_type is self.geometry_type:
                for geometry in item.geometry_objects[self.geometry_type]:
                    self.add(geometry)
            else:
                raise TypeError(
                    "You tried to add a {item.geometry_type} set to a {self.geometry_type} set. "
                    "This is not possible"
                )
        elif self._get_geometry_type(item) is self.geometry_type:
            self.geometry_objects[self.geometry_type][_cast(_Node | _Element, item)] = (
                None
            )
        else:
            raise TypeError(f"Got unexpected geometry type {type(item)}")

    def get_node_dict(self) -> dict[_Node, None]:
        """Determine the explicitly added nodes for this set, i.e., nodes
        contained in elements for element sets are not returned.

        Thus, for non-point sets an empty dict is returned.

        Returns:
            A dictionary containing the explicitly added nodes for this set.
        """
        if self.geometry_type is _bme.geo.point:
            return _cast(dict[_Node, None], self.geometry_objects[_bme.geo.point])
        else:
            return {}

    def get_points(self) -> list[_Node]:
        """Determine all points (represented by nodes) for this set.

        This function only works for point sets.

        Returns:
            A list containing the points (represented by nodes) associated with this set.
        """
        if self.geometry_type is _bme.geo.point:
            return list(self.get_node_dict().keys())
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self) -> list[_Node]:
        """Determine all nodes associated with this set.

        This includes nodes contained within the geometry added to this
        set, e.g., nodes connected to elements in element sets.

        Returns:
            A list containing all associated nodes.
        """

        if self.geometry_type is _bme.geo.point:
            return list(
                _cast(_KeysView[_Node], self.geometry_objects[_bme.geo.point].keys())
            )
        elif self.geometry_type is _bme.geo.line:
            nodes = []
            for element in _cast(
                _KeysView[_Element], self.geometry_objects[_bme.geo.line].keys()
            ):
                nodes.extend(element.nodes)
            # Remove duplicates while preserving order
            return list(dict.fromkeys(nodes))
        else:
            raise TypeError(
                "Currently GeometrySet is only implemented for points and lines"
            )

    def get_geometry_objects(self) -> _Sequence[_Node | _Element]:
        """Get a list of the objects with the specified geometry type.

        Returns:
            A list with the contained geometry.
        """
        return list(self.geometry_objects[self.geometry_type].keys())

    def copy(self) -> "GeometrySet":
        """Create a shallow copy of this object, the reference to the nodes
        will be the same, but the containers storing them will be copied.

        Returns:
            A shallow copy of the geometry set.
        """
        return GeometrySet(list(self.geometry_objects[self.geometry_type].keys()))


class GeometrySetNodes(GeometrySetBase):
    """Geometry set which is defined by nodes and not explicit geometry."""

    def __init__(
        self,
        geometry_type: _conf.Geometry,
        nodes: _Union[_Node, list[_Node], "GeometrySetNodes", None] = None,
        **kwargs,
    ):
        """Initialize the geometry set.

        Args:
            geometry_type: Type of geometry. This is  necessary, as the boundary conditions
                and input file depend on that type.
            nodes: Node(s) or list of nodes to be added to this geometry set.
        """

        if geometry_type not in _bme.geo:
            raise TypeError(f"Expected geometry enum, got {geometry_type}")

        super().__init__(geometry_type, **kwargs)
        self.nodes: dict[_Node, None] = {}
        if nodes is not None:
            self.add(nodes)

    def add(self, value: _Union[_Node, list[_Node], "GeometrySetNodes"]) -> None:
        """Add nodes to this object.

        Args:
            nodes: Node(s) or list of nodes to be added to this geometry set.
        """

        if isinstance(value, list):
            # Loop over items and check if they are either Nodes or integers.
            # This improves the performance considerably when large list of
            # Nodes are added.
            for item in value:
                self.add(item)
        elif isinstance(value, (int, _Node)):
            self.nodes[value] = None
        elif isinstance(value, GeometrySetNodes):
            # Add all nodes from this geometry set.
            if self.geometry_type == value.geometry_type:
                for node in value.nodes:
                    self.add(node)
            else:
                raise TypeError(
                    f"You tried to add a {value.geometry_type} set to a {self.geometry_type} set. "
                    "This is not possible"
                )
        else:
            raise TypeError(f"Expected Node or list, but got {type(value)}")

    def get_node_dict(self) -> dict[_Node, None]:
        """Determine the explicitly added nodes for this set.

        Thus, we can simply return all points here.

        Returns:
            A dictionary containing the explicitly added nodes for this set.
        """
        return self.nodes

    def get_points(self) -> list[_Node]:
        """Determine all points (represented by nodes) for this set.

        This function only works for point sets.

        Returns:
            A list containing the points (represented by nodes) associated with this set.
        """
        if self.geometry_type is _bme.geo.point:
            return list(self.get_node_dict().keys())
        else:
            raise TypeError(
                "The function get_points can only be called for point sets."
                f" The present type is {self.geometry_type}"
            )

    def get_all_nodes(self) -> list[_Node]:
        """Determine all nodes associated with this set.

        This includes nodes contained within the geometry added to this
        set, e.g., nodes connected to elements in element sets.

        Returns:
            A list containing all associated nodes.
        """
        return list(self.get_node_dict().keys())

    def copy(self) -> "GeometrySetNodes":
        """Create a shallow copy of this object, the reference to the nodes
        will be the same, but the containers storing them will be copied.

        Returns:
            A shallow copy of the geometry set.
        """
        return GeometrySetNodes(
            geometry_type=self.geometry_type,
            nodes=list(self.nodes.keys()),
        )


class GeometryName(dict):
    """Group node geometry sets together.

    This is mainly used for export from mesh functions. The sets can be
    accessed by a unique name. There is no distinction between different
    types of geometry, every name can only be used once -> use
    meaningful names.
    """

    def __setitem__(self, key, value):
        """Set a geometry set in this container."""

        if not isinstance(key, str):
            raise TypeError(f"Expected string, got {type(key)}!")
        if isinstance(value, GeometrySetBase):
            super().__setitem__(key, value)
        else:
            raise NotImplementedError("GeometryName can only store GeometrySets")


class GeometrySetContainer(_ContainerBase):
    """A class to group geometry sets together with the key being the geometry
    type."""

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [GeometrySetBase]

        for geometry_key in _bme.geo:
            self[geometry_key] = []

    def copy(self):
        """When creating a copy of this object, all lists in this object will
        be copied also."""

        # Create a new geometry set container.
        copy = GeometrySetContainer()

        # Add a copy of every list from this container to the new one.
        for geometry_key in _bme.geo:
            copy[geometry_key] = self[geometry_key].copy()

        return copy

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
"""This module implements the class that represents one node in the Mesh."""

import numpy as _np

from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem
from beamme.core.rotation import Rotation as _Rotation


class Node(_BaseMeshItem):
    """This object represents one node in the mesh."""

    def __init__(self, coordinates, *, is_middle_node=False, **kwargs):
        super().__init__(**kwargs)

        # Coordinates of this node.
        self.coordinates = _np.array(coordinates)

        # If this node is at the end of a line or curve (by default only those
        # nodes are checked for overlapping nodes).
        self.is_end_node = False

        # If the node is in the middle of a beam element.
        self.is_middle_node = is_middle_node

        # Lists with the objects that this node is linked to.
        self.element_link = []
        self.node_sets_link = []
        self.element_partner_index = None
        self.mesh = None

        # If this node is replaced, store a link to the remaining node.
        self.master_node = None

    def get_master_node(self):
        """Return the master node of this node.

        If the node has not been replaced, this object is returned.
        """

        if self.master_node is None:
            return self
        else:
            return self.master_node.get_master_node()

    def replace_with(self, master_node):
        """Replace this node with another node object."""

        # Check that the two nodes have the same type.
        if not isinstance(self, type(master_node)):
            raise TypeError(
                "A node can only be replaced by a node with the same type. "
                + f"Got {type(self)} and {type(master_node)}"
            )

        # Replace the links to this node in the referenced objects.
        self.mesh.replace_node(self, master_node)
        for element in self.element_link:
            element.replace_node(self, master_node)
        for node_set in self.node_sets_link:
            node_set.replace_node(self, master_node)

        # Set link to master node.
        self.master_node = master_node.get_master_node()

    def unlink(self):
        """Reset the links to elements, node sets and global indices."""
        self.element_link = []
        self.node_sets_link = []
        self.mesh = None
        self.i_global = None

    def rotate(self, *args, **kwargs):
        """Don't do anything for a standard node, as this node can not be
        rotated."""


class NodeCosserat(Node):
    """This object represents a Cosserat node in the mesh, i.e., it contains
    three positions and three rotations."""

    def __init__(
        self,
        coordinates,
        rotation: _Rotation,
        *,
        arc_length: float | None = None,
        **kwargs,
    ):
        super().__init__(coordinates, **kwargs)

        # Rotation of this node.
        self.rotation = rotation.copy()

        # Arc length along the filament that this beam is a part of
        self.arc_length = arc_length

    def rotate(self, rotation, *, origin=None, only_rotate_triads=False):
        """Rotate this node.

        By default the node is rotated around the origin (0,0,0), if the
        keyword argument origin is given, it is rotated around that
        point. If only_rotate_triads is True, then only the rotation is
        affected, the position of the node stays the same.
        """

        self.rotation = rotation * self.rotation

        # Rotate the positions (around origin).
        if not only_rotate_triads:
            if origin is not None:
                self.coordinates = self.coordinates - origin
            self.coordinates = rotation * self.coordinates
            if origin is not None:
                self.coordinates = self.coordinates + origin


class ControlPoint(Node):
    """This object represents a control point with a weight in the mesh."""

    def __init__(self, coordinates, weight, **kwargs):
        super().__init__(coordinates, **kwargs)

        # Weight of this node
        self.weight = weight

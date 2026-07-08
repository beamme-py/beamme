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
"""This module implements the class that represents one node in the Mesh."""

from typing import Any as _Any

import numpy as _np
from numpy.typing import NDArray as _NDArray

from beamme.core.conf import bme as _bme
from beamme.core.rotation import Rotation as _Rotation


class Node:
    """This object represents one node in the mesh."""

    node_type = _bme.node_type.node

    def __init__(self, coordinates, *, is_middle_node=False) -> None:
        # Global index of this node in a mesh.
        self.i_global: None | int = None

        # Coordinates of this node.
        self.coordinates = _np.array(coordinates, dtype=float)

        # If this node is at the end of a line or curve (by default only those
        # nodes are checked for overlapping nodes).
        self.is_end_node = False

        # If the node is in the middle of a beam element.
        self.is_middle_node = is_middle_node

        # Lists with the objects that this node is linked to.
        self.element_link: list[_Any] = []

        # If this node is replaced, store a link to the remaining node.
        self.target_node: "Node" | None = None

    def get_target_node(self) -> "Node":
        """Return the target node of this node.

        Returns:
            If this node has a linked target node, then this target node is returned,
            otherwise this node is returned.
        """
        if self.target_node is None:
            return self
        else:
            return self.target_node.get_target_node()

    def unlink(self) -> None:
        """Reset the links to elements."""
        self.element_link = []


class NodeCosserat(Node):
    """This object represents a Cosserat node in the mesh, i.e., it contains three
    positions and three rotations."""

    node_type = _bme.node_type.cosserat

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

    def rotate(
        self,
        rotation: _Rotation,
        *,
        origin: _NDArray | list[float] | None = None,
        only_rotate_triads: bool = False,
    ) -> None:
        """Rotate this node.

        Args:
            rotation: Rotation that will be applied to this node.
            origin: Point around which the node will be rotated. If None, the
                node will be rotated around the origin (0,0,0).
            only_rotate_triads: If True, only the rotation of this node will be
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

    node_type = _bme.node_type.control_point

    def __init__(self, coordinates, weight, **kwargs):
        super().__init__(coordinates, **kwargs)

        # Weight of this node
        self.weight = weight

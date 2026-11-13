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
"""Define a BeamMe geometry set referencing a geometry in a mesh representation."""

from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.mesh_representation import GeometrySetInfo as _GeometrySetInfo
from beamme.core.mesh_representation import MeshRepresentation as _MeshRepresentation
from beamme.core.node import Node as _Node


class GeometrySetInputFile(_GeometrySetBase):
    """A BeamMe geometry set referencing a geometry in a mesh representation."""

    def __init__(
        self,
        mesh_representation: _MeshRepresentation,
        geometry_set_info: _GeometrySetInfo,
    ):
        super().__init__(geometry_set_info.geometry_type, name=geometry_set_info.name)
        self.geometry_set_id = geometry_set_info.i_global
        self.mesh_representation = mesh_representation

    def get_geometry_set_id(self, mesh_representation: _MeshRepresentation) -> int:
        """Get the geometry set ID.

        Args:
            mesh_representation: The mesh representation where this geometry set should
                be added to, has to be the same object as self.mesh_representation.
                This is used as a check that the geometry set is not added to a
                different mesh representation than the one it was created for.

        Returns:
            The geometry set ID as an integer.
        """
        if mesh_representation is not self.mesh_representation:
            raise ValueError(
                "The provided mesh representation is not the same as the one this"
                "geometry set was created from."
            )
        return self.geometry_set_id

    def check_replaced_nodes(self) -> None:
        """This method is not supported for this class."""
        raise NotImplementedError(
            "`check_replaced_nodes` is not supported for GeometrySetInputFile"
        )

    def get_node_dict(self) -> dict[_Node, None]:
        """This method is not supported for this class."""
        raise NotImplementedError(
            "`get_node_dict` is not supported for GeometrySetInputFile"
        )

    def get_points(self) -> list[_Node]:
        """This method is not supported for this class."""
        raise NotImplementedError(
            "`get_points` is not supported for GeometrySetInputFile"
        )

    def get_all_nodes(self) -> list[_Node]:
        """This method is not supported for this class."""
        raise NotImplementedError(
            "`get_all_nodes` is not supported for GeometrySetInputFile"
        )

    def __add__(self, other):
        """This method is not supported for this class."""
        raise NotImplementedError("`__add__` is not supported for GeometrySetInputFile")

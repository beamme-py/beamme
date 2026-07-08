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
"""This module implements a class to represent boundary conditions."""

import warnings as _warnings
from typing import Dict as _Dict

import beamme.core.conf as _conf
from beamme.core.base_mesh_item import BaseMeshItem as _BaseMeshItem
from beamme.core.conf import bme as _bme
from beamme.core.container import ContainerBase as _ContainerBase
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.utils.nodes import find_close_nodes as _find_close_nodes


class BoundaryConditionBase(_BaseMeshItem):
    """Base class for boundary conditions."""

    def __init__(
        self,
        geometry_set: _GeometrySetBase,
        bc_type: _conf.BoundaryCondition | str,
        **kwargs,
    ):
        """Initialize the boundary condition.

        Args:
            geometry_set: Geometry that this boundary condition acts on.
            bc_type: Type of the boundary condition.
        """
        super().__init__(**kwargs)
        self.bc_type = bc_type
        self.geometry_set = geometry_set


class BoundaryCondition(BoundaryConditionBase):
    """This object represents one boundary condition, e.g., Dirichlet, Neumann, ..."""

    def __init__(
        self,
        geometry_set: _GeometrySetBase,
        data: _Dict,
        bc_type: _conf.BoundaryCondition | str,
        *,
        double_nodes: _conf.DoubleNodes | None = None,
        **kwargs,
    ):
        """Initialize the object.

        Args:
            geometry_set: Geometry that this boundary condition acts on.
            data: Data defining the properties of this boundary condition.
            bc_type: If this is a string, this will be the section that
                this BC will be added to. If it is a bme.bc, the section will
                be determined automatically.
            double_nodes: Depending on this parameter, it will be checked if point
                Neumann conditions do contain nodes at the same spatial positions.
        """
        super().__init__(geometry_set, bc_type, data=data, **kwargs)
        self.double_nodes = double_nodes

        # Perform some sanity checks for this boundary condition.
        self.check()

    def check(self):
        """Check for point Neumann boundaries that there is not a double Node in the
        set.

        Duplicate nodes in a point Neumann boundary condition can lead to the same force
        being applied multiple times at the same spatial position, which results in
        incorrect load application.
        """
        if self.double_nodes is _bme.double_nodes.keep:
            return

        if (
            self.bc_type == _bme.bc.neumann
            and self.geometry_set.geometry_type == _bme.geo.point
        ):
            my_nodes = self.geometry_set.get_points()
            partners = _find_close_nodes(my_nodes)
            # Create a list with nodes that will not be kept in the set.
            double_node_list = []
            for node_list in partners:
                for i, node in enumerate(node_list):
                    if i > 0:
                        double_node_list.append(node)
            if (
                len(double_node_list) > 0
                and self.double_nodes is _bme.double_nodes.remove
            ):
                # Create the a new geometry set with the unique nodes.
                self.geometry_set = _GeometrySet(
                    [node for node in my_nodes if (node not in double_node_list)]
                )
            elif len(double_node_list) > 0:
                _warnings.warn(
                    "There are overlapping nodes in this point Neumann boundary, and it is not "
                    "specified on how to handle them!"
                )


class BoundaryConditionContainer(_ContainerBase):
    """A class to group boundary conditions together.

    The key of the dictionary are (bc_type, geometry_type).
    """

    def __init__(self, *args, **kwargs):
        """Initialize the container and create the default keys in the map."""
        super().__init__(*args, **kwargs)

        self.item_types = [BoundaryConditionBase]

        for bc_key in _bme.bc:
            for geometry_key in _bme.geo:
                self[(bc_key, geometry_key)] = []

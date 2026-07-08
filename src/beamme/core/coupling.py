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
"""This module implements a class to couple geometry together."""

import numpy as _np

import beamme.core.conf as _conf
from beamme.core.boundary_condition import (
    BoundaryConditionBase as _BoundaryConditionBase,
)
from beamme.core.conf import bme as _bme
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.geometry_set import GeometrySetBase as _GeometrySetBase
from beamme.core.node import Node as _Node


class Coupling(_BoundaryConditionBase):
    """Represents a coupling between geometries in 4C."""

    def __init__(
        self,
        geometry: _GeometrySetBase | list[_Node],
        coupling_type: _conf.BoundaryCondition | str,
        coupling_dof_type: _conf.CouplingDofType | dict,
        *,
        check_overlapping_nodes: bool = True,
    ):
        """Initialize this object.

        Args:
            geometry: Geometry set or nodes that should be coupled.
            coupling_type: If this is a string, this will be the section that
                this coupling will be added to. If it is a bme.bc, the section
                will be determined automatically.
            coupling_dof_type: If this is a dictionary it is the dictionary
                that will be used in the input file, otherwise it has to be
                of type bme.coupling_dof.
            check_overlapping_nodes: If all nodes of this coupling condition
                have to be at the same physical position.
        """
        if isinstance(geometry, _GeometrySetBase):
            pass
        elif isinstance(geometry, list):
            geometry = _GeometrySet(geometry)
        else:
            raise TypeError(
                f"Coupling expects a GeometrySetBase item, got {type(geometry)}"
            )

        # Couplings only work for point sets
        if geometry.geometry_type is not _bme.geo.point:
            raise TypeError("Couplings are only implemented for point sets.")

        super().__init__(geometry, bc_type=coupling_type, data=coupling_dof_type)
        self.check_overlapping_nodes = check_overlapping_nodes

        # Perform sanity checks for this boundary condition
        self.check()

    def check(self):
        """Check that all nodes that are coupled have the same position (depending on
        the check_overlapping_nodes parameter)."""
        if not self.check_overlapping_nodes:
            return

        nodes = self.geometry_set.get_points()
        diff = _np.zeros([len(nodes), 3])
        for i, node in enumerate(nodes):
            # Get the difference to the first node
            diff[i, :] = node.coordinates - nodes[0].coordinates
        if _np.max(_np.linalg.norm(diff, axis=1)) > _bme.eps_pos:
            raise ValueError(
                "The nodes given to Coupling do not have the same position."
            )


def coupling_factory(
    geometry: _GeometrySetBase | list[_Node],
    coupling_type: _conf.BoundaryCondition,
    coupling_dof_type: _conf.CouplingDofType | dict,
    **kwargs,
) -> list[Coupling]:
    """Create coupling conditions for the nodes in geometry.

    Args:
        geometry: Geometry set or nodes that should be coupled.
        coupling_type: If this is a string, this will be the section that
            this coupling will be added to. If it is a bme.bc, the section
            will be determined automatically.
        coupling_dof_type: If this is a dictionary it is the dictionary
            that will be used in the input file, otherwise it has to be
            of type bme.coupling_dof.
        kwargs: Will be passed to constructor of `Coupling`.

    Returns:
        A list of coupling objects representing the created coupling conditions.
            - By default, a single coupling object is created that couples all nodes in the given geometry.
            - If the selected coupling type requires pairwise coupling (e.g., due to solver restrictions),
              multiple coupling objects are returned, each coupling a pair of nodes accordingly.
    """
    if not coupling_type.is_point_coupling_pairwise():
        return [Coupling(geometry, coupling_type, coupling_dof_type, **kwargs)]
    else:
        if isinstance(geometry, _GeometrySetBase):
            nodes = geometry.get_points()
        else:
            nodes = geometry
        main_node = nodes[0]
        return [
            Coupling([main_node, node], coupling_type, coupling_dof_type, **kwargs)
            for node in nodes[1:]
        ]

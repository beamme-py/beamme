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
"""This file contains a function to add the beam interaction conditions for 4C."""

import beamme.core.conf as _conf
from beamme.core.boundary_condition import BoundaryCondition as _BoundaryCondition
from beamme.core.geometry_set import GeometrySet as _GeometrySet
from beamme.core.mesh import Mesh as _Mesh


def get_next_possible_id_for_boundary_condition(
    mesh: _Mesh,
    bc_type: _conf.BoundaryCondition,
    geometry_type: _conf.Geometry,
    condition_string: str,
) -> int:
    """Returns the next possible id, which can be used for a boundary condition based on
    all previous added boundary conditions within a mesh.

    It returns the first ID which is not yet occupied within the existing boundary conditions
    w.r.t. to the given search_string and regex group index.

    Args:
        mesh: Mesh containing already set boundary conditions.
        bc_type: Type of the boundary condition to be searched.
        geometry_type: Geometry type of the boundary condition.
        condition_string: String defining the condition ID tag.

    Returns:
        id: Smallest available ID
    """
    found_conditions = []

    # loop through every possible geometry and find the conditions
    if (bc_type, geometry_type) in mesh.boundary_conditions:
        for bc_condition in mesh.boundary_conditions[(bc_type, geometry_type)]:
            found_conditions.append(bc_condition)

    if not found_conditions:
        # return starting id, since no conditions of type has been added.
        return 0
    else:
        existing_ids = []

        # compare string of each condition with input and store existing ids
        for bc in found_conditions:
            if condition_string in bc.data:
                existing_ids.append(bc.data[condition_string])
            else:
                raise KeyError(
                    f"The key {condition_string} is not in the data {bc.data}"
                )

        # return lowest found id
        return min(set(range(len(existing_ids) + 1)) - set(existing_ids))


def add_beam_interaction_condition(
    mesh: _Mesh,
    geometry_set_1: _GeometrySet,
    geometry_set_2: _GeometrySet,
    bc_type: _conf.BoundaryCondition,
    *,
    id: int | None = None,
) -> int:
    """Adds a pair of beam interaction boundary conditions to the given mesh and
    estimates automatically the id of them based on all previously added boundary
    conditions of the mesh.

    Args:
        mesh: Mesh to which the boundary conditions will be added.
        geometry_set_1: GeometrySet 1 for beam interaction boundary condition
        geometry_set_2: GeometrySet 2 for beam interaction boundary condition
        id: id of the two conditions

    Returns:
        id: Used id for the created condition.
    """
    condition_string = "COUPLING_ID"
    if id is None:
        id = get_next_possible_id_for_boundary_condition(
            mesh,
            bc_type,
            geometry_set_1.geometry_type,
            condition_string=condition_string,
        )

        id_2 = get_next_possible_id_for_boundary_condition(
            mesh,
            bc_type,
            geometry_set_2.geometry_type,
            condition_string=condition_string,
        )

        if not id == id_2:
            raise ValueError(
                f"The estimated IDs {id} and {id_2} do not match. Check Inputfile."
            )

    # Creates the two conditions with the same ID.
    for geometry_set in [geometry_set_1, geometry_set_2]:
        mesh.add(
            _BoundaryCondition(
                geometry_set, data={condition_string: id}, bc_type=bc_type
            )
        )

    return id
